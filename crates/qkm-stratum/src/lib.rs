//! Stratum for KawPoW on Quai nodes: the protocol of rs-quai's and go-quai's
//! stratum servers, a client that reconnects and fails over on its own,
//! TLS for pools behind a TLS proxy, and a mock pool that checks shares
//! exactly as rs-quai does.

pub mod client;
pub mod mock;
pub mod proto;
pub mod tls;

pub use client::{Client, Config, Event, Pool, Share, Submitter};
pub use proto::{Job, difficulty_of_target, redact, target_for_difficulty};
pub use tls::{Tls, Trust};
