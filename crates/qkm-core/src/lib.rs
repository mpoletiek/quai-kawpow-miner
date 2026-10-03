//! KawPoW for GPU miners, built on rs-quai's `rsq-pow`: the per-period
//! ProgPoW program, the CUDA kernel sources generated from it, epoch sizes
//! and CPU verification of solutions.

pub mod epoch;
pub mod kernel;
pub mod program;

pub use epoch::{EPOCH_LENGTH, Epoch, Hashes, Verifier, boundary64, meets, period};
pub use program::{Op, Program};
pub use rsq_pow;
