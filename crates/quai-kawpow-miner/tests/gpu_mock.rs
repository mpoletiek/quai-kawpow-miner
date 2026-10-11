//! The miner on the GPU against the mock pool (which checks shares the way
//! rs-quai's stratum does): accepted shares, epoch switches with and
//! without a prefetched DAG, and a stray far-epoch job ignored.
//!
//! Needs the GPU to itself: `cargo test -p quai-kawpow-miner --test gpu_mock
//! -- --ignored --test-threads 1` (LD_LIBRARY_PATH=/opt/cuda/lib64).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qkm_stratum::mock::MockPool;
use qkm_stratum::{Pool, target_for_difficulty};
use quai_kawpow_miner::miner::{MinerConfig, run};
use quai_kawpow_miner::stats::Stats;

fn wait_until(secs: u64, what: &str, mut f: impl FnMut() -> bool) {
    let until = Instant::now() + Duration::from_secs(secs);
    while !f() {
        assert!(Instant::now() < until, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn accepted(s: &Stats) -> u64 {
    s.accepted.load(Ordering::Relaxed)
}

fn job_height(s: &Stats) -> u64 {
    s.json()["job"]["height"].as_u64().unwrap_or(0)
}

fn scenario(prefetch: bool) {
    // About 3 shares a second at 30 MH/s (difficulty 1 = 2^40/255 hashes).
    let pool = MockPool::start(target_for_difficulty(0.002), Some("tok"));
    pool.push_job(7480, true);
    let mut cfg = MinerConfig::new(vec![Pool {
        url: pool.addr(),
        user: "0x00050723a3F8e87f704bb78014FE7791FAf344F5.gpu-test".into(),
        pass: "auth=tok,d=0.002".into(),
        tls: None,
    }]);
    cfg.dag_prefetch = prefetch;
    cfg.status_every = Duration::from_secs(5);
    let stats = Arc::new(Stats::default());
    let stop = Arc::new(AtomicBool::new(false));
    let (s2, st2) = (stats.clone(), stop.clone());
    let miner = std::thread::spawn(move || run(cfg, st2, s2));

    wait_until(120, "10 accepted shares in epoch 0", || {
        accepted(&stats) >= 10
    });
    for h in [7497, 7498, 7499] {
        pool.push_job(h, true);
        std::thread::sleep(Duration::from_millis(700));
    }
    let before = accepted(&stats);
    // The epoch switch.
    pool.push_job(7500, true);
    wait_until(120, "the miner on the epoch 1 job", || {
        job_height(&stats) == 7500
    });
    wait_until(120, "10 accepted shares in epoch 1", || {
        accepted(&stats) >= before + 10
    });
    // One stray job from a far epoch: ignored, then normal work resumes.
    pool.push_job(300_000, true);
    std::thread::sleep(Duration::from_millis(800));
    assert_eq!(job_height(&stats), 7500, "followed a single far-epoch job");
    pool.push_job(7501, true);
    wait_until(30, "the miner on the next normal job", || {
        job_height(&stats) == 7501
    });
    let a = accepted(&stats);
    wait_until(60, "5 more accepted shares", || accepted(&stats) >= a + 5);

    stop.store(true, Ordering::SeqCst);
    miner.join().unwrap().unwrap();
    let t = pool.tally();
    println!("mock tally: {t:?}");
    println!("miner: {}", stats.json());
    assert!(t.rejected.is_empty(), "rejected shares: {:?}", t.rejected);
    assert_eq!(stats.gpu_errors.load(Ordering::Relaxed), 0);
    assert_eq!(stats.rejected.load(Ordering::Relaxed), 0);
}

#[test]
#[ignore = "needs the GPU"]
fn mock_pool_epoch_switch_with_prefetch() {
    scenario(true);
}

#[test]
#[ignore = "needs the GPU"]
fn mock_pool_epoch_switch_rebuild() {
    scenario(false);
}

/// Every GPU in one process: each finds its own shares (disjoint nonce
/// ranges, so the pool sees no duplicates) through one connection.
#[test]
#[ignore = "needs the GPUs"]
fn mock_pool_all_gpus() {
    let n = qkm_cuda::device_count().unwrap();
    let devices: Vec<usize> = (0..n).collect();
    // About 2 shares a second per GPU at 20 MH/s.
    let pool = MockPool::start(target_for_difficulty(0.005), Some("tok"));
    pool.push_job(7480, true);
    let mut cfg = MinerConfig::new(vec![Pool {
        url: pool.addr(),
        user: "0x00050723a3F8e87f704bb78014FE7791FAf344F5.rig-test".into(),
        pass: "auth=tok,d=0.005".into(),
        tls: None,
    }]);
    cfg.devices = devices.clone();
    cfg.status_every = Duration::from_secs(10);
    let stats = Arc::new(Stats::new(&devices));
    let stop = Arc::new(AtomicBool::new(false));
    let (s2, st2) = (stats.clone(), stop.clone());
    let miner = std::thread::spawn(move || run(cfg, st2, s2));

    let per_gpu = |s: &Stats| -> Vec<u64> {
        s.gpus()
            .iter()
            .map(|g| g.accepted.load(Ordering::Relaxed))
            .collect()
    };
    wait_until(300, "5 accepted shares from every GPU", || {
        per_gpu(&stats).iter().all(|&a| a >= 5)
    });
    // New jobs keep every GPU on the same work.
    for h in [7481, 7482, 7483] {
        pool.push_job(h, true);
        std::thread::sleep(Duration::from_secs(3));
    }
    let a = per_gpu(&stats);
    wait_until(120, "3 more shares from every GPU", || {
        per_gpu(&stats).iter().zip(&a).all(|(now, was)| *now >= was + 3)
    });

    stop.store(true, Ordering::SeqCst);
    miner.join().unwrap().unwrap();
    let t = pool.tally();
    println!("mock tally: {t:?}");
    println!("per GPU accepted: {:?}", per_gpu(&stats));
    println!("miner: {}", stats.json());
    assert!(t.rejected.is_empty(), "rejected shares: {:?}", t.rejected);
    assert_eq!(stats.gpu_errors.load(Ordering::Relaxed), 0);
    assert_eq!(stats.rejected.load(Ordering::Relaxed), 0);
    assert_eq!(t.accepted, stats.accepted.load(Ordering::Relaxed));
}

/// Hooks from the running miner: started, connected, a dropped connection,
/// failover when the first pool goes away, and stopping (delivered before
/// `run` returns). Events go to a script that appends them to a file.
#[test]
#[ignore = "needs the GPU"]
fn hooks_from_the_miner() {
    use quai_kawpow_miner::hooks::{HooksConfig, parse_events};
    use std::os::unix::fs::PermissionsExt;

    let dir = std::env::temp_dir().join(format!("qkm-gpu-hooks-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("events");
    let script = dir.join("hook.sh");
    std::fs::write(
        &script,
        format!(
            "#!/bin/sh\necho \"$QKM_EVENT|$QKM_WORKER|$QKM_DETAIL\" >> {}\n",
            out.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let events = || std::fs::read_to_string(&out).unwrap_or_default();
    let count = |name: &str| {
        events()
            .lines()
            .filter(|l| l.starts_with(&format!("{name}|")))
            .count()
    };

    let first = MockPool::start(target_for_difficulty(0.002), Some("tok"));
    let backup = MockPool::start(target_for_difficulty(0.002), Some("tok"));
    first.push_job(7480, true);
    backup.push_job(7480, true);
    let pool = |url: String| Pool {
        url,
        user: "0x00050723a3F8e87f704bb78014FE7791FAf344F5.hooks-test".into(),
        pass: "auth=tok,d=0.002".into(),
        tls: None,
    };
    let mut cfg = MinerConfig::new(vec![pool(first.addr()), pool(backup.addr())]);
    cfg.hooks = HooksConfig {
        command: Some(script.clone()),
        events: Some(parse_events(&["all".into()]).unwrap()),
        interval: Duration::ZERO,
        ..HooksConfig::default()
    };
    let stats = Arc::new(Stats::new(&cfg.devices));
    let stop = Arc::new(AtomicBool::new(false));
    let (s2, st2) = (stats.clone(), stop.clone());
    let miner = std::thread::spawn(move || run(cfg, st2, s2));

    wait_until(120, "3 accepted shares", || accepted(&stats) >= 3);
    wait_until(10, "miner_started and pool_connected", || {
        count("miner_started") == 1 && count("pool_connected") == 1
    });
    first.drop_connections();
    wait_until(30, "a reconnect", || {
        count("pool_disconnected") >= 1 && count("pool_connected") >= 2
    });
    assert_eq!(count("pool_failover"), 0, "{}", events());
    first.stop();
    wait_until(60, "failover to the backup", || count("pool_failover") == 1);
    wait_until(60, "shares on the backup", || backup.tally().accepted >= 2);

    stop.store(true, Ordering::SeqCst);
    miner.join().unwrap().unwrap();
    let all = events();
    println!("{all}");
    let last = all.lines().last().unwrap();
    assert!(
        last.starts_with("miner_stopping|hooks-test|stopping (on request)"),
        "{all}"
    );
    assert!(
        all.contains(&format!("pool_failover|hooks-test|failed over to pool 1, {}", backup.addr())),
        "{all}"
    );
    assert!(!all.contains("auth=tok"), "the token leaked: {all}");
    std::fs::remove_dir_all(&dir).unwrap();
}
