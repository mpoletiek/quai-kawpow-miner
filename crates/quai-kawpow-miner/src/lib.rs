//! quai-kawpow-miner as a library: the stratum miner ([`miner::run`]), its
//! stats, the JSON API, NVML telemetry and GPU tuning.

pub mod api;
pub mod miner;
pub mod stats;
pub mod telemetry;
pub mod tune;
