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

Status: in development. Phase 1 (CUDA kernels with a CPU differential test
and a benchmark) is under way.

## Plan

1. **CUDA core:** per-period ProgPoW kernel generation from `rsq-pow`, DAG
   generation on the GPU, the search kernel, a differential test against
   `rsq-pow` at several epochs, and a benchmark (target: beat kawpowminer's
   30.5 MH/s on an RTX 3060 Ti at epoch 608).
2. **Stratum mining:** a Quai stratum client, job handling, CPU check of
   every share, next-epoch DAG prefetch, failover; tested against rs-quai's
   stratum on mainnet.
3. **Operations:** solo mode over RPC/WebSocket, JSON API, Prometheus,
   NVML, several GPUs, configuration file, packaging.
4. **AMD:** an OpenCL backend.

## License

GPL-3.0-only (it links rs-quai's GPL-3.0 `rsq-pow`).
