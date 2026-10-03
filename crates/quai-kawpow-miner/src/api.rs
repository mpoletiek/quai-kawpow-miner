//! A tiny read-only JSON API: `GET /` (or any path) returns the stats.

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::Duration;

use crate::stats::Stats;

/// Serves the stats on `addr` in a background thread.
pub fn serve(addr: SocketAddr, stats: Arc<Stats>) -> std::io::Result<SocketAddr> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    std::thread::Builder::new()
        .name("api".into())
        .spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                let _ = conn.set_read_timeout(Some(Duration::from_secs(2)));
                let _ = conn.set_write_timeout(Some(Duration::from_secs(2)));
                // Read the request head; the path does not matter.
                let mut r = BufReader::new(match conn.try_clone() {
                    Ok(c) => c,
                    Err(_) => continue,
                });
                let mut line = String::new();
                while r.read_line(&mut line).is_ok_and(|n| n > 2) {
                    line.clear();
                }
                let body = stats.json().to_string();
                let _ = write!(
                    conn,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        })?;
    Ok(bound)
}
