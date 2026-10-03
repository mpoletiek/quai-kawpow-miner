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

Status: phase 1 done (CUDA core); phase 2 (stratum mining) next.

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

## Usage

    export LD_LIBRARY_PATH=/opt/cuda/lib64   # libnvrtc
    quai-kawpow-miner verify                 # GPU vs rsq-pow
    quai-kawpow-miner bench --height 4565235 --seconds 30
    quai-kawpow-miner bench --block 128,256 --grid 256,1824 --streams 2,3   # sweep

Layout: `crates/qkm-core` (program, kernel sources, epochs, CPU
verification), `crates/qkm-cuda` (cudarc backend), `crates/quai-kawpow-miner`
(the binary).

## Plan

1. **CUDA core** (done): per-period ProgPoW kernel generation from
   `rsq-pow`, DAG generation on the GPU, the search kernel, a differential
   test against `rsq-pow` at several epochs, and a benchmark.
2. **Stratum mining:** a Quai stratum client, job handling, CPU check of
   every share, next-epoch DAG prefetch, failover; tested against rs-quai's
   stratum on mainnet.
3. **Operations:** solo mode over RPC/WebSocket, JSON API, Prometheus,
   NVML, several GPUs, configuration file, packaging.
4. **AMD:** an OpenCL backend.

## License

GPL-3.0-only (it links rs-quai's GPL-3.0 `rsq-pow`).
