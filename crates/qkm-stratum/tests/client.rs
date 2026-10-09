//! The client against the mock pool, which checks shares the way rs-quai's
//! stratum does. Shares are mined on the CPU with qkm-kawpow at epoch 0.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::net::TcpListener;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use qkm_core::{Verifier, meets};
use qkm_stratum::mock::{MockJob, MockPool};
use qkm_stratum::{Client, Config, Event, Pool, Share, target_for_difficulty};

const USER: &str = "0x00050723a3F8e87f704bb78014FE7791FAf344F5.test";

/// About one hash in 16 meets it.
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

fn pool(addr: &str, pass: &str) -> Pool {
    Pool {
        url: format!("stratum+tcp://{addr}"),
        user: USER.into(),
        pass: pass.into(),
    }
}

fn quick(pools: Vec<Pool>) -> Config {
    let mut c = Config::new(pools);
    c.connect_timeout = Duration::from_secs(1);
    c.max_backoff = Duration::from_secs(1);
    c
}

/// A nonce (with the 0xc2 extranonce on top) whose hash meets `target`.
fn mine(v: &Verifier, job: &MockJob, target: &[u8; 32], from: u64) -> (u64, [u8; 32]) {
    for i in from.. {
        let nonce = (0xc2u64 << 56) | i;
        let h = v.hash(job.height, &job.header, nonce);
        if meets(&h.hash, target) {
            return (nonce, h.mix);
        }
    }
    unreachable!()
}

fn session_of(events: &[Event]) -> u64 {
    events
        .iter()
        .rev()
        .find_map(|e| match e {
            Event::Connected { session, .. } => Some(*session),
            _ => None,
        })
        .unwrap()
}

#[test]
fn protocol_flow_and_share_checks() {
    let target = easy_target();
    let p = MockPool::start(target, None);
    let job = p.push_job(100, true);
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&p.addr(), "d=0.2")]), tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::Job(_)));
    let session = session_of(&ev);
    assert!(ev.contains(&Event::Extranonce(vec![0xc2])), "{ev:?}");
    assert!(ev.contains(&Event::Authorized), "{ev:?}");
    assert!(ev.contains(&Event::Target(target)), "{ev:?}");
    let Some(Event::Job(j)) = ev.last() else {
        panic!()
    };
    assert_eq!(
        (j.id.as_str(), j.height, j.header),
        (job.id.as_str(), 100, job.header)
    );

    let v = p.verifier();
    let (nonce, mix) = mine(&v, &job, &target, 0);
    let share = Share {
        seq: 1,
        job: job.id.clone(),
        session,
        nonce,
        header: job.header,
        mix,
    };
    c.submit(share.clone());
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::ShareResult { seq: 1, .. }));
    assert!(
        matches!(
            ev.last(),
            Some(Event::ShareResult {
                accepted: true,
                error: None,
                ..
            })
        ),
        "{ev:?}"
    );
    // The same share again, a wrong mix, an unknown job.
    c.submit(Share {
        seq: 2,
        ..share.clone()
    });
    let mut bad_mix = share.clone();
    bad_mix.seq = 3;
    bad_mix.mix[0] ^= 1;
    bad_mix.nonce = mine(&v, &job, &target, (nonce & 0xffff_ffff) + 1).0;
    c.submit(bad_mix);
    c.submit(Share {
        seq: 4,
        job: "nosuchjob".into(),
        ..share.clone()
    });
    let mut errors = std::collections::HashMap::new();
    wait_for(&rx, 10, |e| {
        if let Event::ShareResult {
            seq,
            accepted,
            error,
            ..
        } = e
        {
            assert!(!accepted);
            errors.insert(*seq, error.clone().unwrap());
        }
        errors.len() == 3
    });
    assert_eq!(errors[&2], "duplicate share");
    assert_eq!(errors[&3], "mixhash mismatch");
    assert_eq!(errors[&4], "nojob");
    let t = p.tally();
    assert_eq!(t.accepted, 1);
    assert_eq!(t.workers, vec![USER.to_string()]);
    c.stop();
}

#[test]
fn reconnects_after_the_pool_drops_it() {
    let p = MockPool::start(easy_target(), None);
    p.push_job(100, true);
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&p.addr(), "x")]), tx);
    wait_for(&rx, 10, |e| matches!(e, Event::Job(_)));
    p.drop_connections();
    wait_for(&rx, 10, |e| matches!(e, Event::Disconnected { .. }));
    let ev = wait_for(&rx, 15, |e| matches!(e, Event::Job(_)));
    assert_eq!(session_of(&ev), 2);
    assert!(ev.contains(&Event::Authorized));
    assert_eq!(p.tally().connections, 2);
    c.stop();
}

#[test]
fn fails_over_to_the_next_pool() {
    // A port nobody listens on.
    let dead = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().to_string()
    };
    let p = MockPool::start(easy_target(), None);
    p.push_job(100, true);
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&dead, "x"), pool(&p.addr(), "x")]), tx);
    let ev = wait_for(&rx, 20, |e| matches!(e, Event::Job(_)));
    assert!(
        ev.iter()
            .any(|e| matches!(e, Event::Connected { pool: 1, .. })),
        "{ev:?}"
    );
    c.stop();
}

#[test]
fn auth_token_is_required_and_redacted() {
    let p = MockPool::start(easy_target(), Some("s3cret"));
    p.push_job(100, true);
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&p.addr(), "auth=wrong,d=0.2")]), tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::AuthFailed(_)));
    assert_eq!(ev.last(), Some(&Event::AuthFailed("unauthorized".into())));
    // Nothing the client reports carries the password.
    assert!(!format!("{ev:?}").contains("wrong"));
    c.stop();
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&p.addr(), "auth=s3cret,d=0.2")]), tx);
    wait_for(&rx, 10, |e| matches!(e, Event::Authorized));
    c.stop();
}

#[test]
fn follows_target_changes() {
    let p = MockPool::start(easy_target(), None);
    p.push_job(100, true);
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&p.addr(), "x")]), tx);
    wait_for(&rx, 10, |e| matches!(e, Event::Job(_)));
    let mut t2 = [0xffu8; 32];
    t2[0] = 0x01;
    p.set_target(t2);
    wait_for(&rx, 10, |e| *e == Event::Target(t2));
    p.set_difficulty(0.5);
    wait_for(&rx, 10, |e| *e == Event::Target(target_for_difficulty(0.5)));
    c.stop();
}

#[test]
fn an_unanswered_share_times_out() {
    let p = MockPool::start(easy_target(), None);
    let job = p.push_job(100, true);
    p.set_answer_delay(Duration::from_secs(3));
    let (tx, rx) = channel();
    let mut cfg = quick(vec![pool(&p.addr(), "x")]);
    cfg.response_timeout = Duration::from_millis(800);
    let c = Client::start(cfg, tx);
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::Job(_)));
    c.submit(Share {
        seq: 9,
        job: job.id.clone(),
        session: session_of(&ev),
        nonce: 1,
        header: job.header,
        mix: [0; 32],
    });
    let ev = wait_for(&rx, 10, |e| matches!(e, Event::ShareResult { seq: 9, .. }));
    assert!(
        matches!(ev.last(), Some(Event::ShareResult { error: Some(e), .. }) if e == "no response"),
        "{ev:?}"
    );
    c.stop();
}

#[test]
fn every_pool_refusing_is_fatal() {
    // A Qi/Quai address from another zone (first byte 0x01), then a bad lock.
    let a = MockPool::start(easy_target(), None);
    let b = MockPool::start(easy_target(), None);
    let mut other_zone = pool(&a.addr(), "d=1");
    other_zone.user = "0x01050723a3F8e87f704bb78014FE7791FAf344F5.test".into();
    let bad_lock = pool(&b.addr(), "d=1,lock=7");
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![other_zone, bad_lock]), tx);
    let seen = wait_for(&rx, 20, |e| matches!(e, Event::Fatal(_)));
    let Some(Event::Fatal(why)) = seen.last() else { panic!() };
    assert!(why.contains("not internal to this zone"), "{why}");
    assert!(why.contains("invalid lock value"), "{why}");
    c.stop();
}

#[test]
fn an_unanswered_authorize_reconnects() {
    // Accepts connections and reads, never answers.
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        let mut keep = Vec::new();
        for s in l.incoming().flatten() {
            keep.push(s);
        }
    });
    let (tx, rx) = channel();
    let mut cfg = quick(vec![pool(&addr, "")]);
    cfg.response_timeout = Duration::from_millis(500);
    let c = Client::start(cfg, tx);
    wait_for(&rx, 10, |e| {
        matches!(e, Event::Disconnected { reason } if reason.contains("no answer to mining.subscribe"))
    });
    c.stop();
}

#[test]
fn an_endless_line_ends_the_connection() {
    use std::io::Write;
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap().to_string();
    std::thread::spawn(move || {
        if let Some(Ok(mut s)) = l.incoming().next() {
            let chunk = vec![b'a'; 16 * 1024];
            for _ in 0..8 {
                if s.write_all(&chunk).is_err() {
                    return;
                }
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    });
    let (tx, rx) = channel();
    let c = Client::start(quick(vec![pool(&addr, "")]), tx);
    wait_for(&rx, 10, |e| {
        matches!(e, Event::Disconnected { reason } if reason.contains("longer than 64 KiB"))
    });
    c.stop();
}

#[test]
fn goes_back_to_the_first_pool_once_it_answers() {
    // The first pool is down at the start, then comes up on the same port.
    let spare = TcpListener::bind("127.0.0.1:0").unwrap();
    let first_addr = spare.local_addr().unwrap();
    drop(spare);
    let backup = MockPool::start(easy_target(), None);
    backup.push_job(100, true);
    let (tx, rx) = channel();
    let mut cfg = quick(vec![pool(&first_addr.to_string(), ""), pool(&backup.addr(), "")]);
    cfg.primary_retry = Duration::from_secs(1);
    cfg.probe_every = Duration::from_millis(500);
    let c = Client::start(cfg, tx);
    wait_for(&rx, 20, |e| matches!(e, Event::Connected { pool: 1, .. }));
    // Bring the first pool up where the client expects it.
    let first = TcpListener::bind(first_addr).unwrap();
    let seen = wait_for(&rx, 20, |e| matches!(e, Event::Connected { pool: 0, .. }));
    assert!(seen.iter().any(
        |e| matches!(e, Event::Disconnected { reason } if reason.contains("first pool answers again"))
    ));
    drop(first);
    c.stop();
}
