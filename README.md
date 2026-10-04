# quai-kawpow-miner

A KawPoW GPU miner for Quai Network, written in Rust.

Quai's KawPoW work is merged-mined with Ravencoin: miners hash a Ravencoin
donor header that commits to a Quai block, through a Quai node's stratum
(rs-quai or go-quai) or directly over the node's RPC. This miner is built
for that setup:

- **The node's own algorithm.** KawPoW comes from
  [rs-quai](https://github.com/mpoletiek/rs-quai)'s `rsq-pow`, which is
  verified against go-quai across the whole mainnet chain. The GPU kernels
  are generated from the same code, and every share is checked on the CPU
  before it is sent, so the miner never submits a bad share.
- **Native code for the GPU in hand.** Kernels are compiled for the
  device's architecture at runtime (CUDA through NVRTC to CUBIN), so a CUDA
  toolkit newer than the driver still works.
- **Built to run unattended.** Epoch changes never race a DAG rebuild (the
  next epoch's DAG is prepared ahead), reconnects never crash, and pools
  fail over.
- **Quai's stratum dialect.** `auth=<token>`, `d=<difficulty>`, worker
  names, and the response times a Quai node needs.
- **Solo mining.** Directly against a node's `quai_getBlockTemplate` and
  `quai_submitKawpowBlock`, with no stratum.
- **Observable.** A JSON API, Prometheus metrics and NVML telemetry
  (temperature, power, fan) with limits; several GPUs per host.

Status: phase 2 done (stratum mining on mainnet); phase 3 (operations) next.

## Phase 1 results (RTX 3060 Ti, driver 595.99 / CUDA 13.2, toolkit 13.4)

- **Exact.** `verify` compares the GPU with `rsq-pow` nonce by nonce: 0
  differences in 768 mix digests and final hashes at epochs 0, 300 and 608,
  plus the next period's kernel at each; every solution the search kernel
  reports passes the CPU check. The per-period program itself is tested
  against `rsq_pow::common::prog_loop`.
- **Hashrate.** 30.54 MH/s at epoch 608 (`bench --seconds 30`), against
  kawpowminer's 30.50 MH/s mean run straight after on the same card. Both
  are at the card's limit: memory runs at CUDA's P2 clock (9,251 MHz, about
  592 GB/s), each hash reads 16 KiB of DAG (about 84% of that bandwidth),
  and both sit at the 225 W power cap.
- **DAG.** Built on the GPU in 11.4 s at epoch 608 (kawpowminer: 12.5 s):
  three passes, with the 512-parent loop run by four threads per item so a
  warp reads whole 64-byte cache rows.
- **Kernels.** Compiled at runtime to native code for the device (no PTX),
  one program per period, the next period's compiled in the background.

What the sweeps showed (`bench` takes comma-separated lists): the
read-only data path for DAG loads beats L2-only, streaming and plain loads
by 1-3%; forcing more blocks per SM costs 8-10% (register spills); unrolling
the DAG loop or dropping the compiler fence doesn't help; launch geometry
matters little (block 256, grid 256, two streams is the default); a DAG from
`cuMemAlloc` beats one from the stream-ordered pool.

## Phase 2 results (2026-10-04, mainnet, rs-quai's stratum on the soak node)

- **Live:** 38 minutes on mainnet through rs-quai's stratum (KawPoW
  port, `d=0.2`, worker `qkm0`).
  - 78 shares accepted, against 78.7 expected at 30 MH/s.
  - 0 rejected, stale or lost; 0 disconnects.
  - 30.0 MH/s on the 10-minute average.
  - Submit-to-answer latency 10-22 ms (median 17).
  - The node counted the same 78 valid shares and logged no invalid ones.
- **Coverage:** over 2^20 nonces at epoch 608, the search kernel reports
  exactly the nonces whose final hash meets the target (241 of 241, none
  missing, none extra), per the debug kernel checked against `rsq-pow`. Over
  2^31 nonces at a 1/2^24 target it found 112 solutions (expected 128 +/- 11).
- **End to end against the mock pool,** which checks shares as rs-quai does:
  - 65 shares accepted, 0 rejected;
  - epoch 0 to 1 with a DAG built ahead and ready at the switch, and again
    with a rebuild;
  - a single far-epoch job ignored.
- **Client:** 6 tests against the mock pool:
  - the protocol, and CPU-mined shares checked the rs-quai way, including
    duplicate, mix-mismatch and unknown-job refusals;
  - reconnect after a drop, and failover to a second pool;
  - the auth token, required and never echoed;
  - `set_target` and `set_difficulty`;
  - an unanswered share timing out.

## Usage

    export LD_LIBRARY_PATH=/opt/cuda/lib64   # libnvrtc

    # Mine through a Quai node's stratum (rs-quai or go-quai, KawPoW port 3333).
    quai-kawpow-miner mine --pool stratum+tcp://10.0.0.13:3333 \
        --user 0xYOUR_QUAI_ADDRESS.rig1 --pass d=0.2 \
        --auth-token-file ~/.config/quai/stratum-token --api 127.0.0.1:4073

    quai-kawpow-miner verify                 # GPU vs rsq-pow
    quai-kawpow-miner bench --height 4565235 --seconds 30
    quai-kawpow-miner bench --block 128,256 --grid 256,1824 --streams 2,3   # sweep

`mine`:
- `--pool` takes `host:port` or `stratum+tcp://host:port`; repeat it for
  failover, in order of preference. The miner returns to the first pool
  after ten minutes on a backup.
- `--user` is your payout address plus a worker name.
- `--pass` holds the pool options; `d=0.2` fixes the stratum difficulty (one
  share about every 30 s at 30 MH/s; without it the node's vardiff starts at
  50 and takes minutes to come down).
- `--auth-token-file` reads the stratum token (sent as `auth=<token>`); it is
  never logged and never appears on the command line.
- `--api ADDR` serves the stats as JSON (`curl http://127.0.0.1:4073/`).
- `--config miner.toml` holds the same settings:

      device = 0
      api = "127.0.0.1:4073"
      status_secs = 60

      [[pool]]
      url = "stratum+tcp://10.0.0.13:3333"
      user = "0xYOUR_QUAI_ADDRESS.rig1"
      pass = "d=0.2"
      auth_token_file = "/home/you/.config/quai/stratum-token"

      [[pool]]   # a backup
      url = "stratum+tcp://10.0.0.12:3333"
      user = "0xYOUR_QUAI_ADDRESS.rig1"
      pass = "d=0.2"

How it stays up:
- **Epochs.** The GPU thread alone owns the DAG, so an epoch change can't
  race a build. Near the end of an epoch, the next light cache and the next
  epoch's first kernel are prepared ahead, and the next DAG too when the card
  holds two. A job from a far epoch (a stale template) is followed only when
  a second job confirms it.
- **Shares.** Every GPU solution is checked with `rsq-pow` before it is sent;
  a mismatch is logged as a GPU bug and never submitted.
- **Connections.** Reconnect with backoff, failover, and a reconnect when the
  pool stops answering shares (three in a row) or sends no job for three
  minutes. CUDA errors are retried and never end the miner.

Layout:
- `crates/qkm-core`: program, kernel sources, epochs, CPU verification;
- `crates/qkm-cuda`: the cudarc backend;
- `crates/qkm-stratum`: the stratum client and a mock pool that checks
  shares the way rs-quai's server does;
- `crates/quai-kawpow-miner`: the miner (library) and the binary.

Tests: `cargo test` runs the protocol, client and mock-pool tests on the
CPU. The GPU end-to-end tests (mining through the mock pool across an epoch
change) need the GPU to themselves:

    cargo test --release -p quai-kawpow-miner --test gpu_mock -- --ignored --test-threads 1

## Plan

1. **CUDA core** (done): per-period ProgPoW kernel generation from
   `rsq-pow`, DAG generation on the GPU, the search kernel, a differential
   test against `rsq-pow` at several epochs, and a benchmark.
2. **Stratum mining** (done): a Quai stratum client, job handling, CPU check
   of every share, next-epoch prefetch, failover; tested against rs-quai's
   stratum on mainnet.
3. **Operations:** solo mode over RPC/WebSocket, JSON API, Prometheus,
   NVML, several GPUs, configuration file, packaging.
4. **AMD:** an OpenCL backend.

## License

GPL-3.0-only (it links rs-quai's GPL-3.0 `rsq-pow`).
