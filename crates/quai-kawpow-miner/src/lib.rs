//! quai-kawpow-miner as a library: the stratum miner ([`miner::run`]), its
//! stats, the JSON API, NVML telemetry, GPU tuning and the TUI.

pub mod api;
pub mod hooks;
pub mod miner;
pub mod output;
pub mod quai;
pub mod stats;
pub mod telemetry;
pub mod tui;
pub mod tune;
