# quai-kawpow-miner

A KawPoW GPU miner for Quai Network, written in Rust. NVIDIA (CUDA) for now.
No dev fee.

Quai's KawPoW work is merge-mined with Ravencoin: miners hash a Ravencoin
donor header that commits to a Quai block, through a Quai node's stratum
(rs-quai, or go-quai 0.56 and later). This miner is built for that setup:

- **The node's own algorithm.** KawPoW comes from `crates/qkm-kawpow`,
  ported from [rs-quai](https://github.com/mpoletiek/rs-quai)'s `rsq-pow`,
  which is verified against go-quai across the whole mainnet chain; it
  passes go-quai's Ravencoin block vectors. The GPU kernels are generated
  from the same code, and every share is checked on the CPU before it is
  sent, so the miner never submits a bad share. The miner builds on its
  own: no dependency on rs-quai.
- **Quai, done right.** The payout address is checked (40 hex digits, its
  zone) and the miner says where the rewards go: the Quai or the Qi ledger,
  which zone, and the coinbase lockup (`--lock 0..3`, with its reward
  bonus). The password is checked the way the node parses it. A refusal no
  retry can fix (a wrong zone, lock or token) stops the miner with the
  reason instead of retrying forever.
- **Every GPU, one process.** One stratum connection and one worker for the
  rig; each GPU searches its own part of the nonce space; the light cache
  and the compiled kernels are shared.
- **The whole rig on one screen.** A live view with the hashrate and its
  last ten minutes, shares, the pool and job, power and efficiency, and a
  row per GPU with everything nvidia-smi shows. Or plain log lines.
- **Built to run unattended.** Thermal stop and resume, a stall watchdog,
  device resets after sticky CUDA errors, failover and return to the first
  pool, staggered GPU start, log rotation, and exit codes a supervisor can
  act on. Nothing a pool sends can crash it.
- **GPU tuning built in.** Power limit, locked core and memory clocks, clock
  offsets and fan speed, per GPU, through NVML.
- **Observable.** A JSON API and Prometheus metrics.
- **Native code for the GPU in hand.** Kernels are compiled for the device's
  architecture at runtime (NVRTC to CUBIN), so a CUDA toolkit newer than the
  driver still works.

## Quick start

Requirements: an NVIDIA driver (Linux), and NVRTC from the CUDA toolkit
(`libnvrtc.so`; on Ubuntu the `cuda-nvrtc-13-*` package; elsewhere put the
toolkit's `lib64` on `LD_LIBRARY_PATH`). The miner says so when either is
missing.

    cargo build --release      # target/release/quai-kawpow-miner

    quai-kawpow-miner devices  # what CUDA and NVML see

    # Mine on every GPU through a Quai node's KawPoW stratum.
    quai-kawpow-miner mine --pool stratum+tcp://10.0.0.13:3333 \
        --user 0xYOUR_ADDRESS.rig1 --pass d=1 --lock 0 \
        --auth-token-file ~/.config/quai/stratum-token --api 127.0.0.1:4073

At start it logs where the rewards go, e.g.
`10.0.0.13:3333: paying Qi on cyprus1 (lock 0: the node's default ...)`.

## Quai: payout, ledger and lockup

- **The address picks the ledger.** A Quai address (second byte below
  `0x80`) is paid in Quai, a Qi address (second byte `0x80` or above) in Qi.
  The first byte is the zone (`0x00` is cyprus1); the node refuses an
  address from another zone. The miner checks the address before
  connecting, because the node would quietly pad or cut a short or long one
  and pay someone else.
- **`--lock` sets the coinbase lockup** (sent as `lock=`); a longer lockup
  earns a reward bonus that shrinks over the years:

  | `lock` | Locked for | Bonus (2026) |
  |---|---|---|
  | 0 | the node's default (`--node.coinbase-lockup`, normally 2 weeks) | none |
  | 1 | 3 months | about x1.03 |
  | 2 | 6 months | about x1.085 |
  | 3 | 1 year | about x1.21 |

  It applies to both ledgers. Solo mining over RPC cannot set it, which is
  one reason this miner uses the node's stratum.
- **Password options** are separated by `,` (or `_`): `d=<difficulty>`
  fixes the share difficulty (Quai nodes raise anything below 0.1 to 0.1),
  `frequency=<s>`, `lock=`, and `auth=` (rs-quai's token; the miner adds it
  from `--auth-token-file`, so it never appears on the command line or in
  the log). The miner refuses what the node would refuse or mangle: a `.`
  separator, a lock outside 0..3, a `d` above 1e6 (read as a hash count), or
  a token containing `,` or `_`.
- **Share difficulty.** Aim for a share every 30 s or so: `d=1` suits about
  150 MH/s. The node closes a connection after 10 minutes without a share,
  and the miner warns when `d` makes that likely (or makes so many shares the
  CPU check falls behind). Without `d=` the node's vardiff starts at 50 and
  takes minutes to come down.
- **Accepted shares** count everything the node accepted: shares that only
  prove the miner is alive (`d=`) as well as workshares and blocks. The node
  does not tell them apart in its answer; its stratum HTTP API (port 3336)
  does.

## The screen

`mine` takes over the terminal:

- a header with the worker, where it pays (`Qi · cyprus1 · lock 2`), the
  pool connection and the uptime;
- the total hashrate and its last ten minutes as a graph; shares (accepted,
  rejected, stale, lost; the last one's age), height, epoch and difficulty,
  the pool's answer time, the rig's power and efficiency;
- a row per GPU: state, hashrate, shares and the age of its last one,
  temperature, fan, power against its limit, kH/W, core and memory clocks,
  utilization, VRAM, P-state, PCIe link, and why the clocks are held down
  (power cap, thermal ...). Temperatures turn yellow at 70°C and red at 80°C;
  a GPU in error, failed or too hot shows red. Columns drop by priority on
  narrow terminals;
- the log; shares say which GPU found them.

`q` twice (or Ctrl-C) stops the miner; arrows, PgUp/PgDn and Home/End scroll
the log. `--no-tui`, or a stdout that is not a terminal, prints the log and a
status line instead.

## Options

`mine`:
- `--pool URL` (`host:port` or `stratum+tcp://host:port`), repeated for
  failover in order of preference. kawpowminer's form with the login in the
  URL, `stratum+tcp://0xADDRESS.worker:d=1@host:port` (as in Quai's docs),
  works too; `--user` and `--pass` win over it. After ten minutes on a backup the miner
  probes the first pool every minute and goes back when it answers.
- `--user 0xADDRESS.worker` (the worker name: letters, digits, `-`, `_`).
- `--pass OPTIONS`, `--lock 0..3`, `--auth-token-file FILE`: see above.
- `--devices 0,1,2` (default `all`).
- `--temp-limit C` (default 90; 0 turns it off) and `--temp-resume C`
  (default 15 below): a GPU that reaches the limit pauses until it cools.
- `--start-stagger SECS` (default 5): the n-th GPU builds its DAG n times this
  after the first, so the GPUs do not all change load at once.
- `--report-hashrate`: send the hashrate to the pool every minute
  (`eth_submitHashrate`; Quai nodes log it).
- `--api ADDR`: the HTTP API (see below). Bind it to localhost unless you
  mean to share it.
- `--log-file PATH`: the log and the status lines; it rotates at 50 MiB,
  keeping three old files.
- `--status-secs N`, `--no-dag-prefetch`, `--no-tui`.
- GPU settings: `--power-limit W`, `--lock-core-clocks MIN,MAX`,
  `--lock-mem-clocks MIN,MAX`, `--core-offset MHZ`, `--mem-offset MHZ`,
  `--fan PCT` (see GPU tuning).
- `--config FILE`: all of the above in TOML (flags win):

      devices = [0, 1, 2]        # default: every GPU
      api = "127.0.0.1:4073"
      log_file = "/home/you/qkm/miner.log"
      status_secs = 60
      start_stagger = 5
      lock = 0
      temp_limit = 90
      report_hashrate = false

      [tune]                     # every GPU (applied by `sudo ... tune`)
      power_limit = 200

      [tune.gpu.0]               # device 0's own settings
      power_limit = 150

      [[pool]]
      url = "stratum+tcp://10.0.0.13:3333"
      user = "0xYOUR_ADDRESS.rig1"
      pass = "d=1"
      auth_token_file = "/home/you/.config/quai/stratum-token"

      [[pool]]                   # a backup
      url = "stratum+tcp://10.0.0.12:3333"
      user = "0xYOUR_ADDRESS.rig1"
      pass = "d=1"

  `dist/miner.example.toml` has a commented copy.

Other commands: `devices` (CUDA and NVML's view), `tune` (below), `verify`
(the GPU against the CPU reference, nonce by nonce), `bench` (hashrate, with sweeps
of the launch geometry: `bench --block 128,256 --grid 256,1824 --streams
2,3`), `compile` and `kernel` (the generated CUDA source).

## GPU tuning

    sudo quai-kawpow-miner tune --power-limit 200            # every GPU
    sudo quai-kawpow-miner tune --config miner.toml          # its [tune]
    quai-kawpow-miner tune --show
    sudo quai-kawpow-miner tune --reset

Settings: power limit (W), locked core and memory clocks (MHz), core and
memory clock offsets (MHz) and fan speed (%, which turns automatic fan
control off until reset), for every GPU or per GPU (`[tune.gpu.N]`). They
need root, last until `tune --reset`, a reboot or a driver reload, and are
matched to GPUs by PCI bus id (CUDA and NVML may number GPUs differently).
`mine` applies them too when it runs as root; otherwise it checks the power
limits are as configured and says so in one line.

On an RTX 3060 Ti (7 cards, Quai mainnet, epoch 609), KawPoW is power
bound: 150 W gave 139.7 MH/s for the rig, 175 W 183.8, 200 W 206.2 and
225 W 216.2; 175-200 W was the most efficient (150 kH/W). Locked clocks
matter less than the power limit.

To let an operator account tune without a password, allow only that
command in sudoers (`visudo -f /etc/sudoers.d/qkm-tune`):

    miner ALL=(root) NOPASSWD: /usr/local/bin/quai-kawpow-miner tune *

Running as root, configuration errors give only a line number, never the
file's text.

## Running unattended

`dist/quai-kawpow-miner.service` runs the miner under systemd
(`--no-tui`, restarted on failure). Exit codes:

| Code | Meaning | Restart? |
|---|---|---|
| 0 | stopped on purpose (Ctrl-C, `q`, SIGTERM) | no |
| 1 | a run-time failure: every GPU failed, a GPU stopped hashing (a hung kernel), the stratum client died | yes |
| 2 | needs a fix: bad configuration, missing CUDA libraries, or every pool refused authorization (address, lock, token) | no (`RestartPreventExitStatus=2`) |

What it does on its own:
- **Pools.** Reconnects with backoff; fails over after two failures in a
  row; goes back to the first pool once it answers again. Reconnects when
  the pool stops answering shares (three in a row), sends no job for three
  minutes, or does not answer subscribe or authorize in 10 s. A line over
  64 KiB, bad hex, or a height past any real epoch is refused without harm;
  pool text is stripped of control characters before it is logged.
- **Epochs.** Each GPU's thread alone owns its DAG, so an epoch change cannot
  race a build. Near the end of an epoch the next light cache and kernel
  are prepared ahead, and the next DAG too when the card holds two. A job
  from any epoch other than the current or next one (a stale template) is
  followed only when a second job confirms it.
- **GPUs.** CUDA errors and panics are retried after a pause; after three in
  a row the device's context is reset (an illegal address or ECC fault ruins
  it for good); after eight the GPU is given up. A GPU at the temperature
  limit pauses until it cools. A GPU that stops hashing for two minutes
  ends the miner with code 1, for the supervisor to restart.
- **Shares.** Every GPU solution is checked on the CPU; a mismatch is
  logged as a GPU bug and never sent. The CPU check queue is bounded; drops
  are counted and logged with a hint to raise `d`.

Octominer rigs: without OctOS or HiveOS nothing feeds the fan board's
hardware watchdog, and it resets the rig every 25 minutes; run
[octofan-watchdog](https://github.com/mpoletiek/octofan-watchdog) alongside.

## API and metrics

`--api 127.0.0.1:4073`:
- `GET /`: everything as JSON: hashrate, shares, pool, job, payout, and per
  GPU its state, hashrate, shares, temperature, power, clocks, fan,
  utilization, VRAM, P-state, throttle reasons and PCIe link;
- `GET /metrics`: Prometheus (`qkm_hashrate_hashes_per_second`,
  `qkm_shares_total{result=...}`, `qkm_gpu_temperature_celsius{device,bus,name}`
  and so on);
- `GET /health`: `ok`.

Requests are capped at 8 KiB and 2 s each; there is no CORS header.

## Benchmarks

Measured 2026-10-09 on an Octominer X12 Ultra, six RTX 3060 Ti at a 200 W
power limit each (clocks unlocked, driver 595.99, epoch 609), mining Quai
mainnet through rs-quai's stratum at the node's minimum difficulty (0.1),
one miner at a time, 15 minutes each after 90 s of warm-up.

| Miner | Version | Hashrate (own report) | Refused by the node | Dev fee | Effective |
|---|---|---|---|---|---|
| **quai-kawpow-miner** | 0.1.0 | 186.0 MH/s (two runs: 186.0, 185.7) | 0 | 0% | **186.0 MH/s** |
| Rigel | 1.23.2 | 187.0 MH/s | 0 | 1% | 185.1 MH/s |
| SRBMiner-MULTI | 3.7.3 | 181.7 MH/s | 0 | 0.85% | 180.2 MH/s |
| T-Rex | 0.26.8 | 190.7 MH/s | 11% (duplicate shares) | 1% | 168 MH/s |
| kawpowminer | 1.2.4 | 187.4 MH/s | 11% (duplicate shares) | 0% | 166 MH/s |

- Power-bound at the same 200 W (1,199 W for the six), the kernels are
  within 2.5% of each other. T-Rex's is the fastest raw; quai-kawpow-miner
  ties Rigel and is 2.3% ahead of SRBMiner.
- T-Rex and kawpowminer lose 11% of their shares on a Quai node: the node
  re-sends the same work under a new job id after every accepted share,
  and they search it again from the same nonce, so the node refuses what
  they find again as duplicates. This miner starts every job at a random
  nonce and never sends a share twice.
- The node's own count of valid shares agrees within its statistical error
  (about 5% at this share rate): 206 +/- 10 and 194 +/- 10 MH/s for this
  miner's two runs, 193 +/- 10 for Rigel, 175 +/- 9 for SRBMiner.
- Kernel launch settings (block 128/256/512, grid, streams, DAG load path,
  occupancy, unrolling) change the hashrate by at most 1-2% under a power
  cap, and no setting wins at every power limit; the defaults stay.

## Correctness

- `qkm-kawpow` passes go-quai's KawPoW tests: the Ravencoin block vectors,
  the first KawPoW block and a GPU-mined share
  (`cargo test -p qkm-kawpow --release -- --ignored`).
- `verify` compares the GPU with `qkm-kawpow` nonce by nonce: 0 differences in
  768 mix digests and final hashes at epochs 0, 300 and 608 (plus the next
  period's kernel at each); over 2^20 nonces the search kernel reports
  exactly the nonces whose hash meets the target (241 of 241); over 2^31
  nonces at a 1/2^24 target it found 112 solutions (expected 128 +/- 11).
- Mainnet: 38 minutes on one RTX 3060 Ti through rs-quai's stratum with 78
  shares accepted (78.7 expected), 0 rejected; 7 GPUs from one process with
  0 rejected.
- `cargo test` runs the protocol, client, mock-pool, configuration, API and
  screen tests on the CPU, including hostile pool input. The GPU end-to-end
  tests (mining through a mock pool that checks shares the way rs-quai does,
  across epoch changes, on every GPU) need the GPUs to themselves:

      cargo test --release -p quai-kawpow-miner --test gpu_mock -- --ignored --test-threads 1

Layout: `crates/qkm-core` (program, kernel sources, epochs, CPU
verification), `crates/qkm-kawpow` (KawPoW on the CPU: the reference),
`crates/qkm-cuda` (the cudarc backend), `crates/qkm-stratum`
(the stratum client and mock pool), `crates/quai-kawpow-miner` (the miner,
its screen, API, tuning and the binary).

## Plan

- Solo mining over RPC (`quai_getBlockTemplate` / `quai_submitKawpowBlock`).
- TLS stratum, a HiveOS package, event hooks.
- AMD: an OpenCL backend.

## License

GPL-3.0-only. `crates/qkm-kawpow` is ported from rs-quai's GPL-3.0
`rsq-pow`.
