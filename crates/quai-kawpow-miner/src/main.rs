//! quai-kawpow-miner: KawPoW GPU miner for Quai Network.
//!
//! Phase 1 commands: `verify` (GPU against rs-quai's `rsq-pow`), `bench`
//! (hashrate, with a launch-geometry sweep), `compile` and `kernel`.

use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use qkm_core::kernel::{DagLoad, SearchParams, search_source};
use qkm_core::{Epoch, Program, Verifier, boundary64, meets, period};
use qkm_cuda::{Dag, Geometry, Gpu, Header, Kernels, Searcher, compile_cubin};

/// Ravencoin height at 2026-10-03 (epoch 608).
const DEFAULT_HEIGHT: u64 = 4_565_235;

#[derive(Parser)]
#[command(
    name = "quai-kawpow-miner",
    version,
    about = "KawPoW GPU miner for Quai Network"
)]
struct Cli {
    /// CUDA device index.
    #[arg(long, global = true, default_value_t = 0)]
    device: usize,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compare the GPU's mix digests and final hashes with rsq-pow's, and
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

fn verify(gpu: &Gpu, heights: &[u64], cases: u32, block: u32, seed: u64) -> Result<(), String> {
    let verifier = Verifier::default();
    let mut rng = Rng(seed | 1);
    let mut failures = 0u64;
    for &height in heights {
        let dag = build_dag(gpu, &verifier, height)?;
        let mut kernels = Kernels::new(
            gpu,
            SearchParams {
                block,
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
        // 1) Every nonce: mix digest and final hash against rsq-pow.
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
) -> Result<(), String> {
    let verifier = Verifier::default();
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
                        variants.push(SearchParams {
                            block,
                            min_blocks: mb,
                            dag_load,
                            fence: fence != 0,
                            unroll,
                        });
                    }
                }
            }
        }
    }
    for params in variants {
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
                let (done, _) = searcher
                    .run(&k, &dag, header, 0, 1 << 40, &mut |_, _| {
                        t.elapsed().as_secs_f64() < seconds
                    })
                    .map_err(|e| e.to_string())?;
                let mhs = done as f64 / t.elapsed().as_secs_f64() / 1e6;
                let label = format!(
                    "block {block} grid {grid} streams {s} load {:?} min_blocks {} fence {} unroll {}",
                    params.dag_load, params.min_blocks, params.fence, params.unroll
                );
                println!("{label}: {mhs:.2} MH/s");
                if mhs > best.0 {
                    best = (mhs, label);
                }
            }
        }
    }
    println!("best: {} at {:.2} MH/s", best.1, best.0);
    Ok(())
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.cmd {
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
            let gpu = Gpu::new(cli.device).map_err(|e| e.to_string())?;
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
        } => {
            let gpu = Gpu::new(cli.device).map_err(|e| e.to_string())?;
            println!("{} (sm_{}{})", gpu.name, gpu.arch.0, gpu.arch.1);
            verify(&gpu, &heights, cases, block, seed)
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
        } => {
            let loads = dag_load
                .iter()
                .map(|s| DagLoad::parse(s).ok_or(format!("unknown DAG load {s}")))
                .collect::<Result<Vec<_>, _>>()?;
            let gpu = Gpu::new(cli.device).map_err(|e| e.to_string())?;
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
            )
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
