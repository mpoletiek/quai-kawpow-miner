//! quai-kawpow-miner: KawPoW GPU miner for Quai Network.
//!
//! Commands: `mine` (stratum mining), `verify` (GPU against rs-quai's
//! `qkm-kawpow`), `bench` (hashrate, with a launch-geometry sweep), `compile`
//! and `kernel`.

use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Args, Parser, Subcommand};
use qkm_core::kernel::{DagLoad, SearchParams, search_source};
use qkm_core::{Epoch, Program, Verifier, boundary64, meets, period};
use qkm_cuda::{Dag, Geometry, Gpu, Header, Kernels, Searcher, compile_cubin};
use quai_kawpow_miner::tune::{TuneConfig, Tuning};

/// Ravencoin height at 2026-10-03 (epoch 608).
const DEFAULT_HEIGHT: u64 = 4_565_235;

#[derive(Parser)]
#[command(
    name = "quai-kawpow-miner",
    version,
    about = "KawPoW GPU miner for Quai Network"
)]
struct Cli {
    /// CUDA device index (default 0; for `mine`, the only GPU to use).
    #[arg(long, global = true)]
    device: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

/// How `stratum+ssl://` pools given with `--pool` are checked.
#[derive(Args, Clone, Default)]
struct TlsArgs {
    /// For `stratum+ssl://` pools: accept only the certificate with this
    /// SHA-256 fingerprint (a self-signed one on your TLS proxy), e.g. from
    /// `openssl x509 -noout -fingerprint -sha256 -in cert.pem`.
    #[arg(long, value_name = "SHA256")]
    tls_fingerprint: Option<String>,
    /// For `stratum+ssl://` pools: trust the CAs in this PEM file instead of
    /// Mozilla's.
    #[arg(long, value_name = "PEM", conflicts_with = "tls_fingerprint")]
    tls_ca: Option<std::path::PathBuf>,
}

/// GPU settings (setting them needs root; they last until reset or reboot).
#[derive(Args, Clone, Default)]
struct TuneArgs {
    /// Power limit, watts.
    #[arg(long, value_name = "W")]
    power_limit: Option<u32>,
    /// Lock the core clock to MIN,MAX MHz (or one value for both).
    #[arg(long, value_name = "MIN,MAX", value_delimiter = ',', num_args = 1..=2)]
    lock_core_clocks: Option<Vec<u32>>,
    /// Lock the memory clock to MIN,MAX MHz (or one value for both).
    #[arg(long, value_name = "MIN,MAX", value_delimiter = ',', num_args = 1..=2)]
    lock_mem_clocks: Option<Vec<u32>>,
    /// Core clock offset, MHz.
    #[arg(long, value_name = "MHZ", allow_hyphen_values = true)]
    core_offset: Option<i32>,
    /// Memory clock offset, MHz.
    #[arg(long, value_name = "MHZ", allow_hyphen_values = true)]
    mem_offset: Option<i32>,
    /// Fan speed, percent (turns automatic fan control off until reset).
    #[arg(long, value_name = "PCT")]
    fan: Option<u32>,
}

impl TuneArgs {
    fn tuning(&self) -> Tuning {
        let pair = |v: &Option<Vec<u32>>| v.as_ref().map(|v| [v[0], *v.get(1).unwrap_or(&v[0])]);
        Tuning {
            power_limit: self.power_limit,
            lock_core_clocks: pair(&self.lock_core_clocks),
            lock_mem_clocks: pair(&self.lock_mem_clocks),
            core_offset: self.core_offset,
            mem_offset: self.mem_offset,
            fan: self.fan,
        }
    }
}

// Parsed once; boxing `Mine`'s flags would buy nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Cmd {
    /// Mine through a Quai node's stratum (rs-quai or go-quai).
    Mine {
        /// Pool `host:port`, `stratum+tcp://host:port`, or
        /// `stratum+ssl://host:port` for TLS; repeat for failover, in order
        /// of preference.
        #[arg(long = "pool")]
        pools: Vec<String>,
        /// `0xADDRESS.worker` (the payout address and a worker name).
        #[arg(long)]
        user: Option<String>,
        /// Password options, e.g. `d=0.2` (a fixed stratum difficulty).
        #[arg(long, default_value = "")]
        pass: String,
        /// File holding the stratum auth token (sent as `auth=<token>`;
        /// never logged).
        #[arg(long)]
        auth_token_file: Option<std::path::PathBuf>,
        /// TOML configuration (flags given on the command line win).
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        /// Serve the stats as JSON on this address, e.g. 127.0.0.1:4073.
        #[arg(long)]
        api: Option<std::net::SocketAddr>,
        /// Seconds between status lines.
        #[arg(long)]
        status_secs: Option<u64>,
        /// Never build the next epoch's DAG ahead.
        #[arg(long)]
        no_dag_prefetch: bool,
        /// CUDA devices to mine on, e.g. `0,1,2`, or `all` (the default):
        /// one process, one stratum connection and one worker for them all.
        #[arg(long)]
        devices: Option<String>,
        /// Seconds between one GPU's DAG build and the next GPU's, so the
        /// GPUs don't change load all at once.
        #[arg(long, value_name = "SECS")]
        start_stagger: Option<f64>,
        /// Coinbase lockup (sent as `lock=`): 0 = the node's default
        /// (normally 2 weeks), 1 = 3 months, 2 = 6 months, 3 = 1 year; longer
        /// lockups earn a reward bonus.
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=3))]
        lock: Option<u8>,
        /// Pause a GPU at this temperature, Celsius (default 90; 0 = off).
        #[arg(long, value_name = "C")]
        temp_limit: Option<u32>,
        /// Resume a paused GPU at this temperature (default: limit - 15).
        #[arg(long, value_name = "C")]
        temp_resume: Option<u32>,
        /// Report the hashrate to the pool every minute (eth_submitHashrate).
        #[arg(long)]
        report_hashrate: bool,
        /// Print log lines instead of the full-screen view (the default when
        /// stdout is not a terminal).
        #[arg(long)]
        no_tui: bool,
        /// Append the log and the status lines to this file.
        #[arg(long, value_name = "PATH")]
        log_file: Option<std::path::PathBuf>,
        /// Run this program on events (absolute path; see [hooks] in the
        /// README for the events and its environment).
        #[arg(long, value_name = "PATH")]
        hook_command: Option<std::path::PathBuf>,
        /// POST events to this http:// or https:// URL (Discord, Slack,
        /// ntfy, your own).
        #[arg(long, value_name = "URL")]
        hook_url: Option<String>,
        /// Check the settings (pools, logins, token file, TLS, hooks, GPU
        /// settings) and exit, without touching a GPU: 0 when they are
        /// good, 2 when not.
        #[arg(long)]
        check: bool,
        #[command(flatten)]
        tls: TlsArgs,
        #[command(flatten)]
        tune: TuneArgs,
    },
    /// List the CUDA devices (index, PCI bus, name, memory, power limit).
    Devices,
    /// Show a running miner's full screen (one run as a service, say) from
    /// its API; quitting leaves the miner running.
    Watch {
        /// The miner's API address (default: `api` from --config, else
        /// 127.0.0.1:4073).
        #[arg(long, value_name = "HOST:PORT")]
        api: Option<String>,
        /// Take the address from this configuration's `api`.
        #[arg(long)]
        config: Option<std::path::PathBuf>,
    },
    /// Apply GPU settings (power limit, clocks, fans) and exit; needs root.
    /// Takes the same settings as `mine` (flags or the config's `[tune]`).
    Tune {
        /// CUDA devices, e.g. `0,1,2`, or `all` (the default).
        #[arg(long)]
        devices: Option<String>,
        /// TOML configuration (its `[tune]` section; flags win).
        #[arg(long)]
        config: Option<std::path::PathBuf>,
        /// Print each GPU's current settings instead.
        #[arg(long)]
        show: bool,
        /// Put each GPU back to its defaults instead.
        #[arg(long, conflicts_with = "show")]
        reset: bool,
        #[command(flatten)]
        tune: TuneArgs,
    },
    /// Compare the GPU's mix digests and final hashes with qkm-kawpow's, and
    /// check that every solution the search kernel reports verifies.
    Verify {
        /// Donor heights to test (one DAG each).
        #[arg(long, value_delimiter = ',', default_value = "1,4565235")]
        heights: Vec<u64>,
        /// Nonces compared per height.
        #[arg(long, default_value_t = 256)]
        cases: u32,
        /// Threads per block.
        #[arg(long, default_value_t = 256)]
        block: u32,
        /// Seed for the random headers and nonces.
        #[arg(long, default_value_t = 0x5155_4149)]
        seed: u64,
        /// Check kawpowminer's lane-digest reduction instead of the lean one.
        #[arg(long)]
        classic_digest: bool,
    },
    /// Hashrate for one or more launch geometries (one DAG build).
    Bench {
        /// Donor height.
        #[arg(long, default_value_t = DEFAULT_HEIGHT)]
        height: u64,
        /// Seconds per geometry.
        #[arg(long, default_value_t = 20.0)]
        seconds: f64,
        /// Threads per block (comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "256")]
        block: Vec<u32>,
        /// Blocks per launch (comma-separated to sweep; 0 = 8 waves of full occupancy).
        #[arg(long, value_delimiter = ',', default_value = "256")]
        grid: Vec<u32>,
        /// Streams in flight (comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "2")]
        streams: Vec<usize>,
        /// DAG load path: ldg, cg, cs or plain (comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "ldg")]
        dag_load: Vec<String>,
        /// `__launch_bounds__` minimum blocks per SM, 0 = ptxas default
        /// (comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "0")]
        min_blocks: Vec<u32>,
        /// Compiler fence around the DAG load: 1 or 0 (comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "1")]
        fence: Vec<u8>,
        /// DAG loop unroll factor (comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "1")]
        unroll: Vec<u32>,
        /// Lean lane-digest reduction: 1 (the default) or 0 (kawpowminer's;
        /// comma-separated to sweep).
        #[arg(long, value_delimiter = ',', default_value = "1")]
        lean_digest: Vec<u8>,
        /// Run the whole list this many times (alternating A/B cancels
        /// drift in temperature and clocks).
        #[arg(long, default_value_t = 1)]
        repeat: u32,
    },
    /// Compile the search kernel for a height (NVRTC only; no DAG).
    Compile {
        /// Donor height.
        #[arg(long, default_value_t = DEFAULT_HEIGHT)]
        height: u64,
        /// Threads per block.
        #[arg(long, default_value_t = 256)]
        block: u32,
    },
    /// Print the generated CUDA source for a height.
    Kernel {
        /// Donor height.
        #[arg(long, default_value_t = DEFAULT_HEIGHT)]
        height: u64,
    },
}

/// xorshift64* for test inputs.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn bytes32(&mut self) -> [u8; 32] {
        let mut b = [0u8; 32];
        for c in b.chunks_mut(8) {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        b
    }
}

fn words_le(b: &[u8; 32]) -> [u32; 8] {
    Header::from_bytes(b).0
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn build_dag(gpu: &Gpu, verifier: &Verifier, height: u64) -> Result<Dag, String> {
    let epoch = Epoch::for_height(height);
    let (free, total) = gpu.memory().map_err(|e| e.to_string())?;
    let need = epoch.dataset_bytes + epoch.cache_bytes;
    println!(
        "epoch {}: DAG {:.2} GiB, light cache {:.1} MiB; GPU free {:.2} of {:.2} GiB",
        epoch.number,
        gib(epoch.dataset_bytes),
        epoch.cache_bytes as f64 / (1u64 << 20) as f64,
        gib(free as u64),
        gib(total as u64)
    );
    if (free as u64) < need {
        return Err(format!(
            "not enough free GPU memory for epoch {} ({:.2} GiB needed)",
            epoch.number,
            gib(need)
        ));
    }
    let t = Instant::now();
    let ec = verifier.epoch_cache(height);
    println!("light cache: {:.2} s (CPU)", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let dag = gpu
        .generate_dag(epoch, &ec.cache)
        .map_err(|e| e.to_string())?;
    println!("DAG: {:.2} s (GPU)", t.elapsed().as_secs_f64());
    Ok(dag)
}

fn verify(
    gpu: &Gpu,
    heights: &[u64],
    cases: u32,
    block: u32,
    seed: u64,
    lean_digest: bool,
) -> Result<(), String> {
    let verifier = Verifier::default();
    let mut rng = Rng(seed | 1);
    let mut failures = 0u64;
    for &height in heights {
        let dag = build_dag(gpu, &verifier, height)?;
        let mut kernels = Kernels::new(
            gpu,
            SearchParams {
                block,
                lean_digest,
                ..SearchParams::default()
            },
        );
        let t = Instant::now();
        let k = kernels
            .get(gpu, height, dag.epoch)
            .map_err(|e| e.to_string())?;
        println!(
            "kernel for period {}: {:.2} s",
            period(height),
            t.elapsed().as_secs_f64()
        );
        let searcher = Searcher::new(
            gpu,
            Geometry {
                grid: 64,
                streams: 1,
            },
        )
        .map_err(|e| e.to_string())?;
        // 1) Every nonce: mix digest and final hash against qkm-kawpow.
        let header = rng.bytes32();
        let start = rng.next();
        let got = searcher
            .hash_all(&k, &dag, Header::from_bytes(&header), start, cases)
            .map_err(|e| e.to_string())?;
        let mut bad = 0u32;
        for (i, (mix, fin)) in got.iter().take(cases as usize).enumerate() {
            let nonce = start.wrapping_add(i as u64);
            let want = verifier.hash(height, &header, nonce);
            if words_le(&want.mix) != *mix || words_le(&want.hash) != *fin {
                bad += 1;
                if bad <= 3 {
                    println!(
                        "  MISMATCH nonce {nonce:#018x}: gpu mix {mix:08x?} final {fin:08x?}\n                cpu mix {:08x?} final {:08x?}",
                        words_le(&want.mix),
                        words_le(&want.hash)
                    );
                }
            }
        }
        println!(
            "height {height} (epoch {}, period {}): {cases} nonces, {bad} differ",
            dag.epoch.number,
            period(height)
        );
        failures += u64::from(bad);
        // 1b) The next period (a new program every 3 heights): its kernel
        // was compiled in the background by `get` and must be right too.
        let next = height + 3;
        if Epoch::for_height(next) == dag.epoch {
            let t = Instant::now();
            let k2 = kernels
                .get(gpu, next, dag.epoch)
                .map_err(|e| e.to_string())?;
            let wait = t.elapsed().as_secs_f64();
            let start2 = rng.next();
            let got2 = searcher
                .hash_all(&k2, &dag, Header::from_bytes(&header), start2, 64)
                .map_err(|e| e.to_string())?;
            let bad2 = got2
                .iter()
                .take(64)
                .enumerate()
                .filter(|(i, (mix, fin))| {
                    let want = verifier.hash(next, &header, start2.wrapping_add(*i as u64));
                    words_le(&want.mix) != *mix || words_le(&want.hash) != *fin
                })
                .count();
            println!(
                "  next period {} (prefetched, ready after {wait:.2} s): 64 nonces, {bad2} differ",
                period(next)
            );
            failures += bad2 as u64;
        }
        // 2) The search kernel: solutions against an easy target must verify.
        let mut target = [0xffu8; 32];
        target[0] = 0x00;
        target[1] = 0x3f; // about 1 in 1024 hashes
        let header2 = rng.bytes32();
        let mut searcher = Searcher::new(
            gpu,
            Geometry {
                grid: 16,
                streams: 2,
            },
        )
        .map_err(|e| e.to_string())?;
        let (done, found) = searcher
            .run(
                &k,
                &dag,
                Header::from_bytes(&header2),
                boundary64(&target),
                rng.next(),
                &mut |done, found| done < 1 << 18 && found.len() < 64,
            )
            .map_err(|e| e.to_string())?;
        let mut sol_bad = 0u64;
        for f in &found {
            let want = verifier.hash(height, &header2, f.nonce);
            if words_le(&want.mix) != f.mix || !meets(&want.hash, &target) {
                sol_bad += 1;
            }
        }
        println!(
            "  search: {done} nonces, {} solutions reported, {sol_bad} fail the CPU check",
            found.len()
        );
        if found.is_empty() {
            println!("  search found nothing at a 1/1024 target: FAIL");
            failures += 1;
        }
        failures += sol_bad;
        // 3) Coverage: the search kernel must report exactly the nonces whose
        // final hash meets the target, per the debug kernel (checked against
        // qkm-kawpow above). Small launches keep under MAX_RESULTS per launch.
        let mut cov_target = [0xffu8; 32];
        cov_target[0] = 0x00;
        cov_target[1] = 0x0f; // 1 in 4096
        let header3 = rng.bytes32();
        let start3 = rng.next() & 0x00ff_ffff_ffff_ffff;
        let n3: u32 = 1 << 20;
        let all = searcher
            .hash_all(&k, &dag, Header::from_bytes(&header3), start3, n3)
            .map_err(|e| e.to_string())?;
        let want: std::collections::BTreeSet<u64> = all
            .iter()
            .take(n3 as usize)
            .enumerate()
            .filter(|(_, (_, fin))| {
                let mut b = [0u8; 32];
                for (i, w) in fin.iter().enumerate() {
                    b[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
                }
                meets(&b, &cov_target)
            })
            .map(|(i, _)| start3.wrapping_add(i as u64))
            .collect();
        let mut small = Searcher::new(
            gpu,
            Geometry {
                grid: 4,
                streams: 1,
            },
        )
        .map_err(|e| e.to_string())?;
        let (done3, found3) = small
            .run(
                &k,
                &dag,
                Header::from_bytes(&header3),
                boundary64(&cov_target),
                start3,
                &mut |done, _| done < u64::from(n3),
            )
            .map_err(|e| e.to_string())?;
        let got: std::collections::BTreeSet<u64> = found3
            .iter()
            .map(|f| f.nonce)
            .filter(|n| want.contains(n) || n.wrapping_sub(start3) < u64::from(n3))
            .collect();
        let missing = want.difference(&got).count();
        let extra = got.difference(&want).count();
        println!(
            "  coverage: {done3} nonces, {} qualify (debug kernel), search reported {}, {missing} missing, {extra} extra",
            want.len(),
            got.len()
        );
        failures += (missing + extra) as u64;
        // 4) Rate at scale: solutions per nonce against the target's odds.
        let mut rate_target = [0xffu8; 32];
        rate_target[0] = 0x00;
        rate_target[1] = 0x00;
        rate_target[2] = 0x00; // 1 in 2^24
        let mut big = Searcher::new(
            gpu,
            Geometry {
                grid: 256,
                streams: 2,
            },
        )
        .map_err(|e| e.to_string())?;
        let n4: u64 = 1 << 31;
        let mut count = 0usize;
        let mut done4 = 0u64;
        let mut s4 = rng.next() & 0x00ff_ffff_ffff_ffff;
        let header4 = rng.bytes32();
        while done4 < n4 {
            let (d, f) = big
                .run(
                    &k,
                    &dag,
                    Header::from_bytes(&header4),
                    boundary64(&rate_target),
                    s4,
                    &mut |done, found| done4 + done < n4 && found.is_empty(),
                )
                .map_err(|e| e.to_string())?;
            for x in &f {
                let h = verifier.hash(height, &header4, x.nonce);
                if meets(&h.hash, &rate_target) {
                    count += 1;
                }
            }
            done4 += d;
            s4 = s4.wrapping_add(d);
        }
        let expect = done4 as f64 / f64::from(1u32 << 24);
        println!(
            "  rate: {done4} nonces at 1/2^24: {count} solutions, expected {expect:.0} (+/- {:.0})",
            expect.sqrt()
        );
        if (count as f64 - expect).abs() > 4.0 * expect.sqrt() {
            println!("  rate: outside 4 sigma: FAIL");
            failures += 1;
        }
        drop(small);
        drop(big);
        drop(searcher);
        drop(kernels);
        drop(dag);
        gpu.release_memory().map_err(|e| e.to_string())?;
    }
    if failures == 0 {
        println!("verify: OK");
        Ok(())
    } else {
        Err(format!("verify: {failures} failures"))
    }
}

#[allow(clippy::too_many_arguments)]
fn bench(
    gpu: &Gpu,
    height: u64,
    seconds: f64,
    blocks: &[u32],
    grids: &[u32],
    streams: &[usize],
    loads: &[DagLoad],
    min_blocks: &[u32],
    fences: &[u8],
    unrolls: &[u32],
    leans: &[u8],
    repeat: u32,
) -> Result<(), String> {
    let verifier = Verifier::default();
    // Power for each run (NVML), for the efficiency under a power cap.
    let telemetry = qkm_cuda::pci_bus_id(gpu.ctx.ordinal())
        .ok()
        .and_then(|id| quai_kawpow_miner::telemetry::Telemetry::open(Some(&id), 0));
    let dag = build_dag(gpu, &verifier, height)?;
    let sms = gpu
        .ctx
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
        .map_err(|e| e.to_string())? as u32;
    let mut best = (0.0f64, String::new());
    let mut variants = Vec::new();
    for &block in blocks {
        for &dag_load in loads {
            for &mb in min_blocks {
                for &fence in fences {
                    for &unroll in unrolls {
                        for &lean in leans {
                            variants.push(SearchParams {
                                block,
                                min_blocks: mb,
                                dag_load,
                                fence: fence != 0,
                                unroll,
                                lean_digest: lean != 0,
                            });
                        }
                    }
                }
            }
        }
    }
    let rounds = variants.len() * repeat.max(1) as usize;
    let mut totals: std::collections::HashMap<String, (f64, f64, u32)> =
        std::collections::HashMap::new();
    for params in variants.iter().cycle().take(rounds).copied() {
        let block = params.block;
        let mut kernels = Kernels::new(gpu, params);
        let k = match kernels.get(gpu, height, dag.epoch) {
            Ok(k) => k,
            Err(e) => {
                println!("{params:?}: {e}");
                continue;
            }
        };
        for &g in grids {
            let grid = if g == 0 { sms * 1536 / block * 8 } else { g };
            for &s in streams {
                let mut searcher =
                    Searcher::new(gpu, Geometry { grid, streams: s }).map_err(|e| e.to_string())?;
                let header = Header([0x1234_5678; 8]);
                // Warm-up launch, then the timed run.
                searcher
                    .run(&k, &dag, header, 0, 0, &mut |_, _| false)
                    .map_err(|e| e.to_string())?;
                let t = Instant::now();
                let mut power = (0.0f64, 0u32);
                let mut last = Instant::now();
                let (done, _) = searcher
                    .run(&k, &dag, header, 0, 1 << 40, &mut |_, _| {
                        if last.elapsed() >= Duration::from_millis(500)
                            && let Some(w) = telemetry.as_ref().and_then(|t| t.read().power_w)
                        {
                            last = Instant::now();
                            power = (power.0 + w, power.1 + 1);
                        }
                        t.elapsed().as_secs_f64() < seconds
                    })
                    .map_err(|e| e.to_string())?;
                let mhs = done as f64 / t.elapsed().as_secs_f64() / 1e6;
                let watts = (power.1 > 0).then(|| power.0 / f64::from(power.1));
                let label = format!(
                    "block {block} grid {grid} streams {s} load {:?} min_blocks {} fence {} unroll {} lean {}",
                    params.dag_load,
                    params.min_blocks,
                    params.fence,
                    params.unroll,
                    u8::from(params.lean_digest)
                );
                match watts {
                    Some(w) => println!(
                        "{label}: {mhs:.2} MH/s at {w:.0} W ({:.1} kH/W)",
                        mhs * 1e3 / w
                    ),
                    None => println!("{label}: {mhs:.2} MH/s"),
                }
                let e = totals.entry(label.clone()).or_default();
                *e = (
                    e.0 + mhs,
                    e.1 + watts.map_or(0.0, |w| mhs * 1e3 / w),
                    e.2 + 1,
                );
                if mhs > best.0 {
                    best = (mhs, label);
                }
            }
        }
    }
    if repeat > 1 {
        let mut rows: Vec<_> = totals.into_iter().collect();
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for (label, (mhs, eff, n)) in rows {
            println!(
                "mean of {n}: {label}: {:.2} MH/s, {:.1} kH/W",
                mhs / f64::from(n),
                eff / f64::from(n)
            );
        }
    }
    println!("best: {} at {:.2} MH/s", best.1, best.0);
    Ok(())
}

/// The TOML configuration file.
#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    device: Option<usize>,
    devices: Option<Vec<usize>>,
    api: Option<std::net::SocketAddr>,
    status_secs: Option<u64>,
    dag_prefetch: Option<bool>,
    start_stagger: Option<f64>,
    tui: Option<bool>,
    log_file: Option<std::path::PathBuf>,
    lock: Option<u8>,
    temp_limit: Option<u32>,
    temp_resume: Option<u32>,
    report_hashrate: Option<bool>,
    hooks: Option<FileHooks>,
    #[serde(default)]
    tune: TuneConfig,
    #[serde(default, rename = "pool")]
    pools: Vec<FilePool>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePool {
    url: String,
    user: String,
    #[serde(default)]
    pass: String,
    auth_token_file: Option<std::path::PathBuf>,
    tls_fingerprint: Option<String>,
    tls_ca: Option<std::path::PathBuf>,
}

/// Where `watch` reads the screen: the configured API address (a wildcard
/// one, e.g. 0.0.0.0, through the loopback), or the usual 127.0.0.1:4073.
fn watch_address(api: Option<std::net::SocketAddr>) -> String {
    let Some(mut a) = api else {
        return "127.0.0.1:4073".into();
    };
    if a.ip().is_unspecified() {
        a.set_ip(match a.ip() {
            std::net::IpAddr::V4(_) => std::net::Ipv4Addr::LOCALHOST.into(),
            std::net::IpAddr::V6(_) => std::net::Ipv6Addr::LOCALHOST.into(),
        });
    }
    a.to_string()
}

/// The `[hooks]` section.
#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct FileHooks {
    command: Option<std::path::PathBuf>,
    webhook: Option<String>,
    format: Option<String>,
    events: Option<Vec<String>>,
    hashrate_low_mhs: Option<f64>,
    hashrate_low_mins: Option<f64>,
    pool_down_mins: Option<f64>,
    interval_secs: Option<u64>,
    timeout_secs: Option<u64>,
}

/// The hook settings from `[hooks]` and the flags (which win).
fn hooks_config(
    file: Option<&FileHooks>,
    command: Option<std::path::PathBuf>,
    url: Option<String>,
) -> Result<quai_kawpow_miner::hooks::HooksConfig, String> {
    use quai_kawpow_miner::hooks::{Format, HooksConfig, LowRate, Webhook, parse_events};
    let empty = FileHooks::default();
    let f = file.unwrap_or(&empty);
    let mut cfg = HooksConfig {
        command: command.or_else(|| f.command.clone()),
        ..HooksConfig::default()
    };
    if let Some(c) = &cfg.command {
        if !c.is_absolute() {
            return Err(format!(
                "hooks: the command must be an absolute path, not {}",
                c.display()
            ));
        }
        if !c.is_file() {
            return Err(format!("hooks: no program at {}", c.display()));
        }
    }
    cfg.webhook = url
        .or_else(|| f.webhook.clone())
        .map(|u| Webhook::parse(&u).map_err(|e| format!("hooks: {e}")))
        .transpose()?;
    cfg.format = match f.format.as_deref() {
        None | Some("json") => Format::Json,
        Some("text") => Format::Text,
        Some(o) => return Err(format!("hooks: format is json or text, not {o:?}")),
    };
    if let Some(e) = &f.events {
        cfg.events = Some(parse_events(e).map_err(|e| format!("hooks: {e}"))?);
    }
    let minutes = |v: f64, what: &str| {
        if v.is_finite() && v > 0.0 && v <= 1440.0 {
            Ok(Duration::from_secs_f64(v * 60.0))
        } else {
            Err(format!("hooks: {what} must be between 0 and 1440 minutes"))
        }
    };
    if let Some(mhs) = f.hashrate_low_mhs {
        if !(mhs.is_finite() && mhs > 0.0) {
            return Err("hooks: hashrate_low_mhs must be above 0".into());
        }
        cfg.hashrate_low = Some(LowRate {
            below: mhs * 1e6,
            after: minutes(f.hashrate_low_mins.unwrap_or(5.0), "hashrate_low_mins")?,
        });
    } else if f.hashrate_low_mins.is_some() {
        return Err("hooks: hashrate_low_mins needs hashrate_low_mhs".into());
    }
    if let Some(m) = f.pool_down_mins {
        cfg.pool_down_after = minutes(m, "pool_down_mins")?;
    }
    if let Some(i) = f.interval_secs {
        cfg.interval = Duration::from_secs(i);
    }
    if let Some(t) = f.timeout_secs {
        if !(1..=300).contains(&t) {
            return Err("hooks: timeout_secs must be 1 to 300".into());
        }
        cfg.timeout = Duration::from_secs(t);
    }
    if cfg.is_off() && file.is_some() {
        return Err("hooks: give a command or a webhook".into());
    }
    Ok(cfg)
}

/// TLS for `url`: `None` unless it is a `stratum+ssl://` URL, which is
/// checked against the pinned certificate, the CA file, or Mozilla's CAs.
fn pool_tls(
    url: &str,
    fingerprint: Option<&str>,
    ca: Option<&std::path::Path>,
) -> Result<Option<qkm_stratum::Tls>, String> {
    use qkm_stratum::{Tls, Trust, tls};
    if !tls::wants_tls(url)? {
        if fingerprint.is_some() || ca.is_some() {
            return Err(format!(
                "{url}: tls_fingerprint and tls_ca are for stratum+ssl:// pools"
            ));
        }
        return Ok(None);
    }
    let trust = match (fingerprint, ca) {
        (Some(_), Some(_)) => {
            return Err(format!("{url}: give tls_fingerprint or tls_ca, not both"));
        }
        (Some(f), None) => {
            Trust::Pinned(tls::parse_fingerprint(f).map_err(|e| format!("{url}: {e}"))?)
        }
        (None, Some(c)) => Trust::CaFile(c.to_path_buf()),
        (None, None) => Trust::Public,
    };
    Tls::new(trust).map(Some).map_err(|e| format!("{url}: {e}"))
}

/// A pool URL in kawpowminer's form, `stratum+tcp://USER[:PASS]@host:port`
/// (Quai's docs use it), split into the URL without the login, the user and
/// the password.
fn split_login(url: &str) -> (String, Option<String>, Option<String>) {
    let (scheme, rest) = url.split_once("://").map_or(("", url), |(s, r)| (s, r));
    let Some((login, host)) = rest.rsplit_once('@') else {
        return (url.to_string(), None, None);
    };
    let (user, pass) = login
        .split_once(':')
        .map_or((login, None), |(u, p)| (u, Some(p.to_string())));
    let bare = if scheme.is_empty() {
        host.to_string()
    } else {
        format!("{scheme}://{host}")
    };
    (bare, (!user.is_empty()).then(|| user.to_string()), pass)
}

/// One pool's stratum login, checked: the payout address and worker, and
/// the password (`auth=<token>` from the token file first, the options,
/// then `lock=`). Returns the pool and what is worth telling the user.
fn login(
    url: &str,
    user: &str,
    pass: &str,
    token_file: Option<&std::path::Path>,
    lock: Option<u8>,
    tls: Option<qkm_stratum::Tls>,
) -> Result<(qkm_stratum::Pool, Vec<String>), String> {
    use quai_kawpow_miner::quai;
    let payout = quai::parse_payout(user)?;
    quai::check_worker(user)?;
    let token = match token_file {
        Some(f) => {
            let t = std::fs::read_to_string(f)
                .map_err(|e| format!("cannot read the auth token file {}: {e}", f.display()))?;
            let t = t.trim().to_string();
            if t.is_empty() {
                return Err(format!("the auth token file {} is empty", f.display()));
            }
            Some(t)
        }
        None => None,
    };
    let (pass, notes) = quai::compose_password(pass, token.as_deref(), lock)?;
    let host = qkm_stratum::Pool {
        url: url.to_string(),
        user: user.to_string(),
        pass: String::new(),
        tls: None,
    }
    .host_port()
    .to_string();
    let l = notes.lock.unwrap_or(0);
    let mut msgs = vec![format!(
        "{host}: paying {} on {} (lock {l}: {})",
        payout.ledger,
        payout.zone(),
        quai::lock_meaning(l)
    )];
    if let Some(t) = &tls {
        msgs.push(format!(
            "{host}: TLS, certificate checked against {}",
            t.trust()
        ));
    }
    msgs.extend(notes.warnings.into_iter().map(|w| format!("{host}: {w}")));
    Ok((
        qkm_stratum::Pool {
            url: url.to_string(),
            user: user.to_string(),
            pass,
            tls,
        },
        msgs,
    ))
}

#[allow(clippy::too_many_arguments)]
fn mine(
    device: Option<usize>,
    devices: Option<String>,
    pools: Vec<String>,
    user: Option<String>,
    pass: String,
    auth_token_file: Option<std::path::PathBuf>,
    config: Option<std::path::PathBuf>,
    api: Option<std::net::SocketAddr>,
    status_secs: Option<u64>,
    no_dag_prefetch: bool,
    start_stagger: Option<f64>,
    lock: Option<u8>,
    temp_limit: Option<u32>,
    temp_resume: Option<u32>,
    report_hashrate: bool,
    no_tui: bool,
    log_file: Option<std::path::PathBuf>,
    hook_command: Option<std::path::PathBuf>,
    hook_url: Option<String>,
    check: bool,
    tls_args: TlsArgs,
    tune: Tuning,
) -> Result<(), CliError> {
    use quai_kawpow_miner::miner::{MinerConfig, run as run_miner};
    use quai_kawpow_miner::stats::Stats;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let file = read_config(config.as_deref())?;
    let lock = lock.or(file.lock);
    if lock.is_some_and(|l| l > 3) {
        return Err("lock must be 0, 1, 2 or 3".into());
    }
    let mut notes = Vec::new();
    let mut pool_list = Vec::new();
    if pools.is_empty() {
        for p in &file.pools {
            let (bare, _, _) = split_login(&p.url);
            let tls = pool_tls(&bare, p.tls_fingerprint.as_deref(), p.tls_ca.as_deref())?;
            let (pool, n) = login(
                &bare,
                &p.user,
                &p.pass,
                p.auth_token_file.as_deref(),
                lock,
                tls,
            )?;
            pool_list.push(pool);
            notes.extend(n);
        }
    } else {
        let tls_given = tls_args.tls_fingerprint.is_some() || tls_args.tls_ca.is_some();
        if tls_given
            && !pools
                .iter()
                .any(|u| qkm_stratum::tls::wants_tls(u).unwrap_or(false))
        {
            return Err("--tls-fingerprint and --tls-ca are for stratum+ssl:// pools".into());
        }
        for u in &pools {
            // --user and --pass win over a login in the URL.
            let (bare, url_user, url_pass) = split_login(u);
            let user = user.clone().or(url_user.clone()).ok_or(
                "--user 0xADDRESS.worker is required with --pool (or stratum+tcp://0xADDRESS.worker@host:port)",
            )?;
            let pass = if pass.is_empty() {
                url_pass.clone().unwrap_or_default()
            } else {
                pass.clone()
            };
            // The TLS flags apply to every stratum+ssl:// pool.
            let tls = if qkm_stratum::tls::wants_tls(&bare)? {
                pool_tls(
                    &bare,
                    tls_args.tls_fingerprint.as_deref(),
                    tls_args.tls_ca.as_deref(),
                )?
            } else {
                None
            };
            let (pool, mut n) = login(&bare, &user, &pass, auth_token_file.as_deref(), lock, tls)?;
            if url_user.is_some() && url_user != Some(user.clone()) {
                n.push(format!(
                    "{}: the user in the URL is ignored for --user",
                    pool.host_port()
                ));
            }
            pool_list.push(pool);
            notes.extend(n);
        }
    }
    if pool_list.is_empty() {
        return Err("no pool: give --pool or a [[pool]] in --config".into());
    }
    let mut cfg = MinerConfig::new(pool_list);
    cfg.status_every = Duration::from_secs(status_secs.or(file.status_secs).unwrap_or(30).max(1));
    cfg.dag_prefetch = !no_dag_prefetch && file.dag_prefetch.unwrap_or(true);
    let stagger = start_stagger.or(file.start_stagger).unwrap_or(5.0);
    if !(0.0..=60.0).contains(&stagger) {
        return Err(format!("start stagger {stagger}: 0 to 60 seconds").into());
    }
    cfg.start_stagger = Duration::from_secs_f64(stagger);
    cfg.temp_limit = match temp_limit.or(file.temp_limit).unwrap_or(90) {
        0 => None,
        l => Some(l),
    };
    cfg.temp_resume = temp_resume.or(file.temp_resume);
    if let (Some(l), Some(r)) = (cfg.temp_limit, cfg.temp_resume)
        && r >= l
    {
        return Err(format!("the resume temperature {r}C must be below the limit {l}C").into());
    }
    cfg.report_hashrate = report_hashrate || file.report_hashrate.unwrap_or(false);
    cfg.tune = file.tune.with(&tune);
    cfg.hooks = hooks_config(file.hooks.as_ref(), hook_command, hook_url)?;
    cfg.tune.check()?;
    if check {
        for n in &notes {
            println!("{n}");
        }
        if !cfg.hooks.is_off() {
            println!("{}", cfg.hooks.describe());
        }
        println!("configuration OK");
        return Ok(());
    }
    qkm_cuda::libraries().map_err(|e| e.to_string())?;
    cfg.devices = mine_devices(devices.as_deref(), device, &file)?;
    let stats = Arc::new(Stats::new(&cfg.devices));
    let stop = Arc::new(AtomicBool::new(false));
    let s = stop.clone();
    ctrlc::set_handler(move || s.store(true, Ordering::SeqCst))
        .map_err(|e| format!("signal handler: {e}"))?;
    let log_file = log_file.or(file.log_file);
    // The log file first: a bad path must fail before the screen is taken.
    quai_kawpow_miner::output::init(log_file.as_deref()).map_err(|e| {
        let p = log_file.as_deref().map(|p| p.display().to_string());
        format!("log file {}: {e}", p.unwrap_or_default())
    })?;
    let want_tui =
        !no_tui && file.tui.unwrap_or(true) && std::io::IsTerminal::is_terminal(&std::io::stdout());
    let tui = if want_tui {
        match quai_kawpow_miner::tui::start(stats.clone(), stop.clone(), log_file.as_deref()) {
            Ok(h) => {
                quai_kawpow_miner::output::set_tui(true);
                Some(h)
            }
            Err(e) => {
                eprintln!("cannot start the full-screen view ({e}); printing log lines");
                None
            }
        }
    } else {
        None
    };
    for n in notes {
        quai_kawpow_miner::output::event(format!("{} {n}", quai_kawpow_miner::miner::now_hms()));
    }
    if let Some(addr) = api.or(file.api) {
        match quai_kawpow_miner::api::serve(addr, stats.clone()) {
            Ok(bound) => quai_kawpow_miner::output::event(format!(
                "{} API: http://{bound}/",
                quai_kawpow_miner::miner::now_hms()
            )),
            Err(e) => {
                stop.store(true, Ordering::SeqCst);
                if let Some(h) = tui {
                    let _ = h.join();
                }
                return Err(format!("API on {addr}: {e}").into());
            }
        }
    }
    let r = run_miner(cfg, stop.clone(), stats.clone()).map_err(|f| {
        if f.permanent {
            CliError::Config(f.reason)
        } else {
            CliError::Runtime(f.reason)
        }
    });
    stop.store(true, Ordering::SeqCst);
    if let Some(h) = tui {
        // A TUI stuck on a dead terminal must not keep the process alive.
        let until = Instant::now() + Duration::from_secs(5);
        while !h.is_finished() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        if h.is_finished() {
            let _ = h.join();
        }
        // The screen is gone: leave the totals behind.
        println!("{} {}", quai_kawpow_miner::miner::now_hms(), stats.line());
        for l in stats.gpu_lines() {
            println!("{l}");
        }
    }
    r
}

/// Running as root (`sudo ... tune`).
fn is_root() -> bool {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() == 0 }
}

fn read_config(path: Option<&std::path::Path>) -> Result<FileConfig, String> {
    let Some(p) = path else {
        return Ok(FileConfig::default());
    };
    let text =
        std::fs::read_to_string(p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
    toml::from_str(&text).map_err(|e| {
        // As root, never echo the file: `sudo ... --config /etc/shadow`
        // would print what the user may not read.
        if is_root() {
            let at = e.span().map_or(String::new(), |s| {
                let line = text[..s.start.min(text.len())].matches('\n').count() + 1;
                format!(" at line {line}")
            });
            format!(
                "{}: not a valid configuration{at} (details hidden when running as root)",
                p.display()
            )
        } else {
            format!("{}: {e}", p.display())
        }
    })
}

/// `tune`: apply, show or reset the GPU settings.
fn tune(
    devices: Option<String>,
    device: Option<usize>,
    config: Option<std::path::PathBuf>,
    show: bool,
    reset: bool,
    flags: Tuning,
) -> Result<(), String> {
    let file = read_config(config.as_deref())?;
    let devices = mine_devices(devices.as_deref(), device, &file)?;
    let tc = file.tune.with(&flags);
    tc.check()?;
    if !show && !reset && tc.is_empty() {
        return Err("nothing to set: give settings (see --help), --show or --reset".into());
    }
    let nvml = nvml_wrapper::Nvml::init().map_err(|e| format!("NVML: {e}"))?;
    let mut failed = false;
    for d in devices {
        let id = qkm_cuda::pci_bus_id(d).map_err(|e| e.to_string())?;
        let (line, ok) = if show {
            (quai_kawpow_miner::tune::show(&nvml, &id), true)
        } else if reset {
            quai_kawpow_miner::tune::reset(&nvml, &id)
        } else {
            let t = tc.for_device(d);
            if t.is_empty() {
                continue;
            }
            let a = quai_kawpow_miner::tune::apply(&nvml, &id, &t);
            (a.text, a.failed == 0)
        };
        failed |= !ok;
        println!("gpu{d} ({id}): {line}");
    }
    if failed {
        return Err("some settings were not applied (see above)".into());
    }
    Ok(())
}

/// The GPUs to mine on: `--devices`, else `--device`, else the config's
/// `devices` or `device`, else every GPU.
fn mine_devices(
    devices: Option<&str>,
    device: Option<usize>,
    file: &FileConfig,
) -> Result<Vec<usize>, String> {
    let list = match devices {
        Some(d) if d.trim() != "all" => parse_devices(d)?,
        Some(_) => Vec::new(),
        None => match (device, &file.devices, file.device) {
            (Some(d), ..) => vec![d],
            (None, Some(ds), _) => {
                let mut seen = ds.clone();
                seen.sort_unstable();
                if seen.windows(2).any(|w| w[0] == w[1]) {
                    return Err(format!("devices {ds:?}: a device is listed twice"));
                }
                ds.clone()
            }
            (None, None, Some(d)) => vec![d],
            (None, None, None) => Vec::new(),
        },
    };
    let count = qkm_cuda::device_count().map_err(|e| e.to_string())?;
    if count == 0 {
        return Err("no CUDA device".into());
    }
    if list.is_empty() {
        return Ok((0..count).collect());
    }
    if let Some(d) = list.iter().find(|&&d| d >= count) {
        return Err(format!("no CUDA device {d} (there are {count})"));
    }
    Ok(list)
}

/// `0,1,2` → [0, 1, 2], refusing repeats.
fn parse_devices(s: &str) -> Result<Vec<usize>, String> {
    let mut out: Vec<usize> = Vec::new();
    for p in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let d = p
            .parse()
            .map_err(|_| format!("bad device {p:?} in --devices (want e.g. 0,1,2 or all)"))?;
        if out.contains(&d) {
            return Err(format!("device {d} given twice in --devices"));
        }
        out.push(d);
    }
    if out.is_empty() {
        return Err("--devices is empty".into());
    }
    Ok(out)
}

/// `devices`: what CUDA and NVML see.
fn list_devices() -> Result<(), String> {
    qkm_cuda::libraries().map_err(|e| e.to_string())?;
    let n = qkm_cuda::device_count().map_err(|e| e.to_string())?;
    let nvml = nvml_wrapper::Nvml::init().ok();
    if nvml.is_none() {
        println!("(NVML unavailable: no power or temperature)");
    }
    for d in 0..n {
        let bus = qkm_cuda::pci_bus_id(d).unwrap_or_else(|_| "?".into());
        let line = match Gpu::new(d) {
            Ok(g) => {
                let total = g.memory().map_or(0, |m| m.1) as f64 / f64::from(1u32 << 30);
                format!("{} (sm_{}{}), {total:.2} GiB", g.name, g.arch.0, g.arch.1)
            }
            Err(e) => format!("cannot open: {e}"),
        };
        let extra = nvml
            .as_ref()
            .and_then(|n| n.device_by_pci_bus_id(bus.as_str()).ok())
            .map(|dev| {
                let t = dev
                    .temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu)
                    .map_or("-".into(), |t| format!("{t}C"));
                let pl = dev
                    .power_management_limit()
                    .map_or("-".into(), |p| format!("{} W", p / 1000));
                format!(", {t}, power limit {pl}")
            })
            .unwrap_or_default();
        println!("{d}: {bus}  {line}{extra}");
    }
    Ok(())
}

/// How a command failed: exit code 2 when only a fix helps (configuration,
/// a refusal from every pool), 1 otherwise.
#[derive(Debug)]
enum CliError {
    Config(String),
    Runtime(String),
}

impl From<String> for CliError {
    fn from(e: String) -> Self {
        CliError::Config(e)
    }
}

impl From<&str> for CliError {
    fn from(e: &str) -> Self {
        CliError::Config(e.to_string())
    }
}

fn run(cli: Cli) -> Result<(), CliError> {
    let device = cli.device.unwrap_or(0);
    if matches!(
        cli.cmd,
        Cmd::Verify { .. } | Cmd::Bench { .. } | Cmd::Compile { .. } | Cmd::Tune { .. }
    ) {
        qkm_cuda::libraries().map_err(|e| e.to_string())?;
    }
    let r: Result<(), String> = match cli.cmd {
        Cmd::Mine {
            pools,
            user,
            pass,
            auth_token_file,
            config,
            api,
            status_secs,
            no_dag_prefetch,
            devices,
            start_stagger,
            lock,
            temp_limit,
            temp_resume,
            report_hashrate,
            no_tui,
            log_file,
            hook_command,
            hook_url,
            check,
            tls,
            tune: t,
        } => {
            return mine(
                cli.device,
                devices,
                pools,
                user,
                pass,
                auth_token_file,
                config,
                api,
                status_secs,
                no_dag_prefetch,
                start_stagger,
                lock,
                temp_limit,
                temp_resume,
                report_hashrate,
                no_tui,
                log_file,
                hook_command,
                hook_url,
                check,
                tls,
                t.tuning(),
            );
        }
        Cmd::Devices => list_devices(),
        Cmd::Watch { api, config } => {
            let addr = match api {
                Some(a) => a,
                None => watch_address(read_config(config.as_deref())?.api),
            };
            quai_kawpow_miner::tui::watch(&addr).map_err(|e| e.to_string())
        }
        Cmd::Tune {
            devices,
            config,
            show,
            reset,
            tune: t,
        } => tune(devices, cli.device, config, show, reset, t.tuning()),
        Cmd::Kernel { height } => {
            let e = Epoch::for_height(height);
            print!(
                "{}",
                search_source(
                    &Program::new(period(height)),
                    e.dag_elements(),
                    SearchParams::default()
                )
            );
            Ok(())
        }
        Cmd::Compile { height, block } => {
            let gpu = Gpu::new(device).map_err(|e| e.to_string())?;
            let e = Epoch::for_height(height);
            let src = search_source(
                &Program::new(period(height)),
                e.dag_elements(),
                SearchParams {
                    block,
                    ..SearchParams::default()
                },
            );
            let t = Instant::now();
            let cubin = compile_cubin(&src, gpu.arch).map_err(|e| e.to_string())?;
            println!(
                "{} (sm_{}{}): search kernel for period {} compiled in {:.2} s, {} bytes",
                gpu.name,
                gpu.arch.0,
                gpu.arch.1,
                period(height),
                t.elapsed().as_secs_f64(),
                cubin.len()
            );
            gpu.load(cubin).map_err(|e| e.to_string())?;
            let t = Instant::now();
            let dag =
                compile_cubin(qkm_core::kernel::DAG_SOURCE, gpu.arch).map_err(|e| e.to_string())?;
            gpu.load(dag).map_err(|e| e.to_string())?;
            println!("DAG kernel compiled in {:.2} s", t.elapsed().as_secs_f64());
            Ok(())
        }
        Cmd::Verify {
            heights,
            cases,
            block,
            seed,
            classic_digest,
        } => {
            let gpu = Gpu::new(device).map_err(|e| e.to_string())?;
            println!("{} (sm_{}{})", gpu.name, gpu.arch.0, gpu.arch.1);
            verify(&gpu, &heights, cases, block, seed, !classic_digest)
        }
        Cmd::Bench {
            height,
            seconds,
            block,
            grid,
            streams,
            dag_load,
            min_blocks,
            fence,
            unroll,
            lean_digest,
            repeat,
        } => {
            let loads = dag_load
                .iter()
                .map(|s| DagLoad::parse(s).ok_or(format!("unknown DAG load {s}")))
                .collect::<Result<Vec<_>, _>>()?;
            let gpu = Gpu::new(device).map_err(|e| e.to_string())?;
            println!("{} (sm_{}{})", gpu.name, gpu.arch.0, gpu.arch.1);
            bench(
                &gpu,
                height,
                seconds,
                &block,
                &grid,
                &streams,
                &loads,
                &min_blocks,
                &fence,
                &unroll,
                &lean_digest,
                repeat,
            )
        }
    };
    r.map_err(CliError::Runtime)
}

/// Exit codes: 0 stopped on purpose; 1 failed at run time (a supervisor
/// should restart it); 2 a configuration problem or a refusal from every
/// pool (restarting will not help).
fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Runtime(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
        Err(CliError::Config(e)) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logins_in_urls() {
        assert_eq!(
            split_login("stratum+tcp://0xab.rig:d=1,lock=2@10.0.0.13:3333"),
            (
                "stratum+tcp://10.0.0.13:3333".into(),
                Some("0xab.rig".into()),
                Some("d=1,lock=2".into())
            )
        );
        assert_eq!(
            split_login("0xab.rig@node:3335"),
            ("node:3335".into(), Some("0xab.rig".into()), None)
        );
        assert_eq!(
            split_login("stratum+tcp://node:3333"),
            ("stratum+tcp://node:3333".into(), None, None)
        );
    }

    #[test]
    fn tls_from_the_url() {
        let fp = "AB:".repeat(31) + "AB";
        assert_eq!(pool_tls("node:3333", None, None), Ok(None));
        assert!(pool_tls("stratum+tcp://node:3333", Some(&fp), None).is_err());
        let t = pool_tls("stratum+ssl://node:3333", None, None)
            .unwrap()
            .unwrap();
        assert_eq!(t.trust(), &qkm_stratum::Trust::Public);
        let t = pool_tls("stratum+tls://node:3333", Some(&fp), None)
            .unwrap()
            .unwrap();
        assert_eq!(t.trust(), &qkm_stratum::Trust::Pinned([0xab; 32]));
        assert!(pool_tls("stratum+ssl://node:3333", Some("ab"), None).is_err());
        assert!(pool_tls("stratum+ssl://n:1", Some(&fp), Some("/x.pem".as_ref())).is_err());
        assert!(pool_tls("stratum+ssl://n:1", None, Some("/nonexistent.pem".as_ref())).is_err());
        assert!(pool_tls("http://node:3333", None, None).is_err());
    }

    /// The shipped example parses, with and without its comments taken out
    /// (every commented setting is valid too).
    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../dist/miner.example.toml");
        let f: FileConfig = toml::from_str(text).unwrap();
        assert_eq!(f.pools.len(), 1);
        let uncommented: String = text
            .lines()
            .map(|l| {
                // `#key = value` and `#[section]`, not prose.
                l.strip_prefix('#')
                    .filter(|rest| {
                        rest.starts_with('[')
                            || rest.split_once(" = ").is_some_and(|(k, _)| {
                                !k.is_empty()
                                    && k.chars()
                                        .all(|c| c.is_ascii_lowercase() || c == '_' || c == '.')
                            })
                    })
                    .unwrap_or(l)
            })
            .collect::<Vec<_>>()
            .join("\n");
        let f: FileConfig = toml::from_str(&uncommented).unwrap();
        assert_eq!(f.pools.len(), 2);
        let h = f.hooks.unwrap();
        assert_eq!(h.hashrate_low_mhs, Some(180.0));
    }

    #[test]
    fn hook_settings() {
        use quai_kawpow_miner::hooks::{Format, Kind};
        let off = hooks_config(None, None, None).unwrap();
        assert!(off.is_off());
        let f: FileConfig = toml::from_str(
            r#"
            [hooks]
            webhook = "https://ntfy.sh/rig"
            format = "text"
            events = ["default", "share_rejected"]
            hashrate_low_mhs = 180
            pool_down_mins = 3
            interval_secs = 30
            "#,
        )
        .unwrap();
        let c = hooks_config(f.hooks.as_ref(), None, None).unwrap();
        assert_eq!(c.format, Format::Text);
        assert!(c.wants(Kind::ShareRejected) && c.wants(Kind::GpuFailed));
        assert!(!c.wants(Kind::PoolConnected));
        assert_eq!(c.hashrate_low.unwrap().below, 180e6);
        assert_eq!(c.hashrate_low.unwrap().after, Duration::from_secs(300));
        assert_eq!(c.pool_down_after, Duration::from_secs(180));
        assert_eq!(c.interval, Duration::from_secs(30));
        // The flag wins.
        let c = hooks_config(f.hooks.as_ref(), None, Some("http://10.0.0.11:9/x".into())).unwrap();
        assert!(
            c.describe().contains("http://10.0.0.11/…"),
            "{}",
            c.describe()
        );
        let bad = |t: &str| {
            let f: FileConfig = toml::from_str(t).unwrap();
            hooks_config(f.hooks.as_ref(), None, None).unwrap_err()
        };
        assert!(bad("[hooks]\nevents = [\"gpu_failed\"]").contains("command or a webhook"));
        assert!(bad("[hooks]\ncommand = \"rel/x.sh\"").contains("absolute"));
        assert!(bad("[hooks]\ncommand = \"/nonexistent/x.sh\"").contains("no program"));
        assert!(bad("[hooks]\nwebhook = \"ftp://x\"").contains("http"));
        assert!(bad("[hooks]\nwebhook = \"http://x\"\nevents = [\"nope\"]").contains("unknown"));
        assert!(bad("[hooks]\nwebhook = \"http://x\"\nformat = \"xml\"").contains("json or text"));
        assert!(bad("[hooks]\nwebhook = \"http://x\"\nhashrate_low_mins = 3").contains("needs"));
        assert!(toml::from_str::<FileConfig>("[hooks]\ncomand = \"/x\"").is_err());
    }

    #[test]
    fn watch_addresses() {
        assert_eq!(watch_address(None), "127.0.0.1:4073");
        assert_eq!(
            watch_address(Some("10.0.0.14:4080".parse().unwrap())),
            "10.0.0.14:4080"
        );
        assert_eq!(
            watch_address(Some("0.0.0.0:4073".parse().unwrap())),
            "127.0.0.1:4073"
        );
        assert_eq!(
            watch_address(Some("[::]:4073".parse().unwrap())),
            "[::1]:4073"
        );
    }

    #[test]
    fn exit_codes() {
        assert!(matches!(CliError::from("x"), CliError::Config(_)));
    }
}
