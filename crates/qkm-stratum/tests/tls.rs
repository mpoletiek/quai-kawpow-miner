//! The client over TLS: the mock pool behind a TLS proxy (as a node behind
//! stunnel or nginx would be), with a self-signed certificate (pinned), a
//! certificate from a private CA (CA file), and the ways a TLS connection
//! is refused.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use qkm_core::meets;
use qkm_stratum::mock::MockPool;
use qkm_stratum::tls::{fingerprint, fingerprint_text};
use qkm_stratum::{Client, Config, Event, Pool, Share, Tls, Trust};
use rcgen::{
    BasicConstraints, CertificateParams, CertifiedIssuer, CertifiedKey, IsCa, KeyPair,
    generate_simple_self_signed,
};
use rustls::ServerConnection;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

const USER: &str = "0x00050723a3F8e87f704bb78014FE7791FAf344F5.tls";

fn easy_target() -> [u8; 32] {
    let mut t = [0xffu8; 32];
    t[0] = 0x0f;
    t
}

fn wait_for(rx: &Receiver<Event>, secs: u64, mut want: impl FnMut(&Event) -> bool) -> Vec<Event> {
    let until = Instant::now() + Duration::from_secs(secs);
    let mut seen = Vec::new();
    loop {
        let left = until.saturating_duration_since(Instant::now());
        assert!(!left.is_zero(), "timed out; events so far: {seen:?}");
        if let Ok(e) = rx.recv_timeout(left) {
            let hit = want(&e);
            seen.push(e);
            if hit {
                return seen;
            }
        }
    }
}

fn quick(pools: Vec<Pool>) -> Config {
    let mut c = Config::new(pools);
    c.connect_timeout = Duration::from_secs(1);
    c.max_backoff = Duration::from_secs(1);
    c
}

fn tls_pool(url: String, trust: Trust) -> Pool {
    Pool {
        url,
        user: USER.into(),
        pass: "auth=tok,d=0.2".into(),
        tls: Some(Tls::new(trust).unwrap()),
    }
}

fn key_der(k: &KeyPair) -> PrivateKeyDer<'static> {
    PrivateKeyDer::Pkcs8(k.serialize_der().into())
}

/// A TLS terminator in front of `upstream`: decrypts what the client sends
/// and forwards it, encrypts what comes back.
fn proxy(upstream: String, cert: CertificateDer<'static>, key: PrivateKeyDer<'static>) -> u16 {
    let cfg = Arc::new(
        rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .unwrap(),
    );
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for down in listener.incoming().flatten() {
            let (cfg, upstream) = (cfg.clone(), upstream.clone());
            std::thread::spawn(move || pump(cfg, down, &upstream));
        }
    });
    port
}

fn idle(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

fn pump(cfg: Arc<rustls::ServerConfig>, mut down: TcpStream, upstream: &str) {
    // Like stunnel: the pool hears of a client only after its handshake.
    let mut tls = ServerConnection::new(cfg).unwrap();
    down.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    while tls.is_handshaking() {
        if tls.complete_io(&mut down).is_err() {
            let _ = down.shutdown(Shutdown::Both);
            return;
        }
    }
    let Ok(mut up) = TcpStream::connect(upstream) else {
        return;
    };
    let tick = Some(Duration::from_millis(2));
    down.set_read_timeout(tick).unwrap();
    up.set_read_timeout(tick).unwrap();
    let mut buf = [0u8; 4096];
    'outer: loop {
        match tls.read_tls(&mut down) {
            Ok(0) => break,
            Ok(_) => {
                if tls.process_new_packets().is_err() {
                    let _ = tls.write_tls(&mut down);
                    break;
                }
            }
            Err(e) if idle(&e) => {}
            Err(_) => break,
        }
        loop {
            match tls.reader().read(&mut buf) {
                Ok(0) => break 'outer,
                Ok(n) => {
                    if up.write_all(&buf[..n]).is_err() {
                        break 'outer;
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(_) => break 'outer,
            }
        }
        match up.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if tls.writer().write_all(&buf[..n]).is_err() {
                    break;
                }
            }
            Err(e) if idle(&e) => {}
            Err(_) => break,
        }
        while tls.wants_write() {
            if tls.write_tls(&mut down).is_err() {
                break 'outer;
            }
        }
    }
    tls.send_close_notify();
    let _ = tls.write_tls(&mut down);
    let _ = down.shutdown(Shutdown::Both);
    let _ = up.shutdown(Shutdown::Both);
}

fn self_signed() -> (CertificateDer<'static>, PrivateKeyDer<'static>) {
    let CertifiedKey { cert, signing_key } =
        generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()]).unwrap();
    (cert.der().clone(), key_der(&signing_key))
}

fn connected(ev: &[Event]) -> (u64, bool) {
    ev.iter()
        .rev()
        .find_map(|e| match e {
            Event::Connected { session, tls, .. } => Some((*session, *tls)),
            _ => None,
        })
        .unwrap()
}

#[test]
fn mines_through_a_tls_proxy_with_a_pinned_certificate() {
    let target = easy_target();
    let p = MockPool::start(target, Some("tok"));
    let job = p.push_job(100, true);
    let (cert, key) = self_signed();
    let pin = fingerprint(&cert);
    let port = proxy(p.addr(), cert, key);

    let (tx, rx) = channel();
    let pool = tls_pool(
        format!("stratum+ssl://127.0.0.1:{port}"),
        Trust::Pinned(pin),
    );
    let c = Client::start(quick(vec![pool]), tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::Job(_)));
    let (session, tls) = connected(&ev);
    assert!(tls, "{ev:?}");
    assert!(ev.contains(&Event::Authorized), "{ev:?}");

    let v = p.verifier();
    let nonce = (0u64..)
        .map(|i| (0xc2u64 << 56) | i)
        .find(|&n| meets(&v.hash(job.height, &job.header, n).hash, &target))
        .unwrap();
    let mix = v.hash(job.height, &job.header, nonce).mix;
    c.submit(Share {
        seq: 1,
        job: job.id.clone(),
        session,
        nonce,
        header: job.header,
        mix,
    });
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::ShareResult { seq: 1, .. }));
    assert!(
        matches!(ev.last(), Some(Event::ShareResult { accepted: true, .. })),
        "{ev:?}"
    );
    assert_eq!(p.tally().accepted, 1);

    // The pool drops the connection: back again, over TLS.
    p.drop_connections();
    let ev = wait_for(&rx, 15, |e| matches!(e, Event::Job(_)));
    assert!(
        ev.iter().any(
            |e| matches!(e, Event::Disconnected { reason } if reason.contains("closed by the pool"))
        ),
        "{ev:?}"
    );
    assert_eq!(connected(&ev), (2, true));
    c.stop();
}

#[test]
fn a_certificate_from_a_private_ca() {
    let p = MockPool::start(easy_target(), Some("tok"));
    p.push_job(100, true);
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate().unwrap()).unwrap();
    let leaf_key = KeyPair::generate().unwrap();
    let leaf = CertificateParams::new(vec!["localhost".to_string()])
        .unwrap()
        .signed_by(&leaf_key, &ca)
        .unwrap();
    let port = proxy(p.addr(), leaf.der().clone(), key_der(&leaf_key));
    let ca_file = std::env::temp_dir().join(format!("qkm-tls-ca-{}.pem", std::process::id()));
    std::fs::write(&ca_file, ca.as_ref().pem()).unwrap();

    let (tx, rx) = channel();
    let pool = tls_pool(
        format!("stratum+ssl://localhost:{port}"),
        Trust::CaFile(ca_file.clone()),
    );
    let c = Client::start(quick(vec![pool]), tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::Job(_)));
    assert!(ev.contains(&Event::Authorized), "{ev:?}");
    c.stop();

    // The same certificate under another name is refused.
    let (tx, rx) = channel();
    let pool = tls_pool(
        format!("stratum+ssl://127.0.0.1:{port}"),
        Trust::CaFile(ca_file.clone()),
    );
    let c = Client::start(quick(vec![pool]), tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::Disconnected { .. }));
    let Some(Event::Disconnected { reason }) = ev.last() else {
        panic!()
    };
    assert!(reason.contains("not valid for name"), "{reason}");
    c.stop();
    let _ = std::fs::remove_file(ca_file);
}

/// The reason for the first failed connection.
fn refusal(pool: Pool) -> String {
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool]), tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::Disconnected { .. }));
    c.stop();
    assert!(!ev.contains(&Event::Authorized), "{ev:?}");
    assert!(
        !ev.iter().any(|e| matches!(e, Event::Connected { .. })),
        "{ev:?}"
    );
    let Some(Event::Disconnected { reason }) = ev.last() else {
        panic!()
    };
    reason.clone()
}

#[test]
fn refusals_name_the_certificate_to_pin() {
    let p = MockPool::start(easy_target(), Some("tok"));
    p.push_job(100, true);
    let (cert, key) = self_signed();
    let actual = fingerprint_text(&fingerprint(&cert));
    let port = proxy(p.addr(), cert, key);
    let url = format!("stratum+ssl://127.0.0.1:{port}");

    // Self-signed, checked against Mozilla's CAs.
    let r = refusal(tls_pool(url.clone(), Trust::Public));
    assert!(r.contains("UnknownIssuer"), "{r}");
    assert!(r.contains(&actual), "{r}");
    assert!(r.contains("tls_fingerprint"), "{r}");

    // Pinned to another certificate.
    let r = refusal(tls_pool(url, Trust::Pinned([7u8; 32])));
    assert!(r.contains(&actual), "{r}");
    assert_eq!(p.tally().connections, 0);
}

#[test]
fn a_plain_pool_under_a_tls_url_is_refused() {
    let p = MockPool::start(easy_target(), Some("tok"));
    p.push_job(100, true);
    let started = Instant::now();
    let r = refusal(tls_pool(
        format!("stratum+ssl://{}", p.addr()),
        Trust::Pinned([7u8; 32]),
    ));
    assert!(r.contains("TLS handshake"), "{r}");
    assert!(started.elapsed() < Duration::from_secs(5));
}
