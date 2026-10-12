# Contributing

Bug reports, fixes and improvements are welcome. For anything larger than a
small fix, open an issue first, so we can agree on the approach before you
spend time on it. Report security problems privately (see
[SECURITY.md](SECURITY.md)).

## Building

You need Rust (the version is pinned in `rust-toolchain.toml`; rustup
installs it on the first `cargo` command) and a C compiler. Building doesn't
need CUDA: the miner loads the NVIDIA driver and NVRTC when it runs.

    cargo build --release

[README.md](README.md#correctness-and-tests) describes the layout of the
crates.

## Before you open a pull request

CI runs these on every pull request; run them first:

    cargo fmt --all
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test --workspace

`cargo test` runs everything that works without a GPU: the KawPoW
reference, the stratum client against a mock pool (including hostile input
and TLS), configuration, alerts, the API and the screen. Add tests with your
change: a fix comes with a test that fails without it.

## Testing on a GPU

Changes to the kernels, the GPU code or the mining loop also need testing on
an NVIDIA GPU (8 GB or more, with the driver and NVRTC; see the README's
requirements):

    # The GPU against the CPU reference, hash by hash, at several epochs:
    cargo run --release -p quai-kawpow-miner -- verify --heights 1,2250000,4577070

    # Mining end to end through a mock pool that checks shares like a node
    # (needs the GPUs to itself):
    cargo test --release -p quai-kawpow-miner --test gpu_mock -- --ignored --test-threads 1

`verify` must report `0 differ`, `0 missing, 0 extra` and `verify: OK`.

For performance changes, compare with `bench` on the same card at a fixed
power limit, alternating the old and new kernels so temperature drift
cancels out (`--repeat 3`), and report MH/s and kH/W at the power limits you
tried. Measurements without a gain are worth recording too: the module
documentation of `crates/qkm-core/src/kernel.rs` lists what has already
been tried.

## Style

- Match the code around you: its naming, its comment density, how errors
  are reported.
- Comments say why, not what. Messages the user sees say what happened and
  what to do about it.
- Keep a pull request to one change, with a commit message that explains
  it.
- Keep the README in step with behavior the user sees: options, messages,
  defaults.

## License

By contributing, you agree that your contributions are licensed under the
project's license, GPL-3.0-only.
