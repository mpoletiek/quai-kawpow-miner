//! KawPoW for GPU miners, built on `qkm-kawpow` (ported from rs-quai's `qkm-kawpow`): the per-period
//! ProgPoW program, the CUDA kernel sources generated from it, epoch sizes
//! and CPU verification of solutions.

pub mod epoch;
pub mod kernel;
pub mod program;

pub use epoch::{
    EPOCH_LENGTH, Epoch, Hashes, MAX_EPOCH, Verifier, boundary64, meets, period, plausible_height,
};
pub use program::{Op, Program};
pub use qkm_kawpow;
