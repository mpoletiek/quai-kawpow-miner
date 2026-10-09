//! go-quai's KawPoW tests (`consensus/kawpow/kawpow_test.go` and
//! `real_blocks_test.go`), as rs-quai's `rsq-pow` ports them, without
//! rs-quai's types: the Ravencoin block vectors carry their header hash.
//!
//! The hash vectors build an epoch's light cache (37-66 MB) and are
//! `#[ignore]`d: `cargo test -p qkm-kawpow --release -- --ignored`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use qkm_kawpow::EpochCaches;
use qkm_kawpow::kawpow::{self, Kawpow};

fn unhex(s: &str) -> Vec<u8> {
    let s = s.trim_start_matches("0x");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Go's `HexToHash`: 32 bytes, left-padded.
fn h32(s: &str) -> [u8; 32] {
    let b = unhex(s);
    let mut out = [0u8; 32];
    out[32 - b.len()..].copy_from_slice(&b);
    out
}

fn light(caches: &EpochCaches, height: u64, hash: &[u8; 32], nonce: u64) -> ([u8; 32], [u8; 32]) {
    let ec = caches.get(height);
    kawpow::kawpow_light(
        &ec,
        kawpow::PARAMS.dataset_size(height),
        hash,
        nonce,
        height,
    )
}

#[test]
fn go_rejects_oversized_donor_height_before_cache_allocation() {
    for h in [75_000_000u64, 750_000_000, u64::from(u32::MAX)] {
        assert!(kawpow::validate_block_number(h).is_err(), "{h}");
        assert!(
            Kawpow::default().compute_light(&[0; 32], 0, h).is_err(),
            "{h}"
        );
    }
    assert!(kawpow::validate_block_number(u64::MAX).is_err());
    kawpow::validate_block_number(1_219_737).unwrap();
}

#[test]
fn go_kawpow_constants_and_activation() {
    assert_eq!(kawpow::PARAMS.epoch_length, 7500);
    assert_eq!(qkm_kawpow::common::CACHE_WORDS, 4096);
    assert_eq!((1_219_736 + 1) / kawpow::PARAMS.epoch_length, 162);
    for (epoch, height) in [(162, 1_219_737), (163, 1_222_500), (200, 1_500_000)] {
        assert_eq!(height / kawpow::PARAMS.epoch_length, epoch);
    }
}

#[test]
fn ethash_sizes() {
    // Ethash's tables at epoch 0, and epoch 609 (October 2026), whose DAG
    // the miner builds on the GPU (24,150,014 256-byte elements, rounded
    // down).
    let p = kawpow::PARAMS;
    assert_eq!(p.cache_size(1), 16_776_896);
    assert_eq!(p.dataset_size(1), 1_073_739_904);
    let ds = p.dataset_size(609 * 7500 + 1);
    assert_eq!(ds, 6_182_403_712);
    assert_eq!(ds / 256, 24_150_014);
}

#[test]
#[ignore = "builds the epoch-162 KawPoW cache"]
fn go_kawpow_first_ravencoin_block_and_gpu_share() {
    let caches = EpochCaches::new(kawpow::PARAMS, 2);
    // First Ravencoin KAWPOW block: Go tries the header hash in both byte
    // orders and requires one of them to give the stored mix hash.
    let want = unhex("b0e016aaa82b7bc95a1ff881d5b42038307fe36429510fcf61659a53e47dd5a8");
    let mixes: Vec<Vec<u8>> = [
        "e7946fbc0ddbf37a6210ae87ea66387621a39a8987e176d0c3574e9deb247aa2",
        "a27a24eb9d4e57c3d076e187899aa321763866ea87ae10627af3db0dbc6f94e7",
    ]
    .iter()
    .map(|hh| light(&caches, 1_219_737, &h32(hh), 0xe9d8d6).0.to_vec())
    .collect();
    assert!(mixes.contains(&want), "{mixes:x?}");
    // "GPU Mined block validation" (height 1219736, same epoch).
    let (mix, _) = light(
        &caches,
        1_219_736,
        &h32("c09b76e782c9c43795599f0e3f204ea87efab090800055547621301b0f5fa899"),
        281_475_099_654_374,
    );
    assert_eq!(
        mix.to_vec(),
        unhex("e1837a2b25364d04a5b05458a066d236a747e047bdebbf12512130a9c271afc5")
    );
}

#[test]
#[ignore = "builds the epoch-396 KawPoW cache"]
fn go_ravencoin_kawpow_vectors() {
    let vectors: serde_json::Value =
        serde_json::from_str(include_str!("data/ravencoin_kawpow_vectors.json")).unwrap();
    let caches = EpochCaches::new(kawpow::PARAMS, 1);
    let num = |v: &serde_json::Value| {
        u64::from_str_radix(v.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
    };
    let vectors = vectors.as_array().unwrap();
    assert!(!vectors.is_empty());
    for v in vectors {
        let name = v["description"].as_str().unwrap();
        let height = v["height"].as_u64().unwrap();
        let hh = h32(v["headerHash"].as_str().unwrap());
        let (mix, _) = light(&caches, height, &hh, num(&v["nonce"]));
        assert_eq!(
            mix,
            h32(v["expectedMixHash"].as_str().unwrap()),
            "{name}: mix hash"
        );
    }
}
