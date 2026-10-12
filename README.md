# quai-kawpow-miner

A KawPoW GPU miner for Quai Network, written in Rust, for NVIDIA GPUs on
Linux. No dev fee.

It mines through a Quai node's KawPoW stratum (go-quai 0.55.1 or later, or
rs-quai), checks every share on the CPU before sending it, and runs every GPU
in a rig from one process. It can run in a terminal with a full-screen view,
or as a systemd service that mines from boot, with alerts, a JSON API and
Prometheus metrics.

**Contents**

- [What you need](#what-you-need)
- [Quick start](#quick-start)
- [Setting up the node](#setting-up-the-node)
- [Configuration](#configuration)
- [Running in a terminal](#running-in-a-terminal)
- [Running as a service](#running-as-a-service)
- [Monitoring and alerts](#monitoring-and-alerts)
- [TLS](#tls)
- [GPU tuning](#gpu-tuning)
- [Octominer rigs](#octominer-rigs)
- [Troubleshooting](#troubleshooting)
- [Command reference](#command-reference)
- [Building from source](#building-from-source)
- [How it works](#how-it-works)
- [Benchmarks](#benchmarks)
- [Correctness and tests](#correctness-and-tests)
- [Roadmap](#roadmap) · [License](#license)

## What you need

- **NVIDIA GPUs with 8 GB or more**, Turing (RTX 20, GTX 16) or newer. The
  DAG is 5.8 GB at epoch 610 and grows by 8 MiB an epoch (about every five
  days), so 6 GB cards no longer fit. Tested on the RTX 3060 Ti.
- **Linux on x86-64** with glibc 2.28 or later, and systemd to run it as a
  service.
- **The NVIDIA driver** (580 or later, for CUDA 13) and **NVRTC**, CUDA's
  runtime compiler (`libnvrtc.so.13`). The miner compiles its kernels for
  your GPU when it starts; it does not need the rest of the CUDA toolkit.
- **A Quai node with its KawPoW stratum on**, reachable from the rig (see
  [Setting up the node](#setting-up-the-node)).
- **A payout address in the node's zone**: a Quai address is paid in Quai,
  a Qi address in Qi.

## Quick start

These steps take a fresh Ubuntu 26.04 machine to mining. Package names on
other versions and distributions differ; what matters is the driver and
`libnvrtc.so.13` on the library path.

**1. Install the driver and NVRTC, then reboot.**

    sudo apt update
    sudo apt install nvidia-headless-595-open nvidia-utils-595 cuda-nvrtc-13-1
    echo /usr/local/cuda-13.1/targets/x86_64-linux/lib | sudo tee /etc/ld.so.conf.d/cuda-13-1-nvrtc.conf
    sudo ldconfig
    sudo reboot

After the reboot, `nvidia-smi` lists your GPUs.

**2. Get the miner.** Download `quai-kawpow-miner-0.1.0-linux-x86_64.tar.gz`
and `SHA256SUMS` from the
[releases page](https://github.com/mpoletiek/quai-kawpow-miner/releases),
then:

    sha256sum -c SHA256SUMS
    tar -xzf quai-kawpow-miner-0.1.0-linux-x86_64.tar.gz
    cd quai-kawpow-miner-0.1.0-linux-x86_64
    ./quai-kawpow-miner devices          # the GPUs it sees

(Or [build it from source](#building-from-source).)

**3. Try it in the terminal.** Replace the node address, your payout
address and the worker name (`rig1`, your name for this machine):

    ./quai-kawpow-miner mine --pool stratum+tcp://192.168.1.10:3333 \
        --user 0xYOUR_ADDRESS.rig1 --pass d=1

`d=` is the share difficulty: use about your rig's hashrate in MH/s divided
by 150 (0.2 for one RTX 3060 Ti, 1.5 for seven), and no less than 0.1. If
the node is rs-quai with an auth token, put the token in a file and add
`--auth-token-file /path/to/token`.

At start it logs where the rewards go, e.g. `paying Qi on cyprus1 (lock 0:
the node's default ...)`. Then the screen fills in: each GPU builds its DAG
(about 10 s each), hashes, and shares start coming in. `q` twice stops it.

**4. Install it as a service** that mines from boot. Copy the example
config, set the same values in its `[[pool]]` section, and install:

    cp miner.example.toml miner.toml
    nano miner.toml
    sudo ./install.sh --config miner.toml --start

`install.sh` checks the config before it enables anything, and says what's
wrong if it isn't right. With an rs-quai token, also uncomment
`auth_token_file` in `miner.toml` and add `--token /path/to/token`;
`install.sh` installs the token where that line points
(`/etc/quai-kawpow-miner/stratum-token`), readable only by root and the
miner's user.

**5. Watch it.**

    quai-kawpow-miner watch --config /etc/quai-kawpow-miner/miner.toml

That is the same full screen, read from the running service; `q` closes it
and the miner keeps mining. On an Octominer, also install
[octofan-watchdog](#octominer-rigs), or the rig resets every 25 minutes.

## Setting up the node

The miner talks to the stratum server built into the Quai node. It is off
by default.

**go-quai** (v0.55.1 or later): start the node with

    --node.stratum-enabled

The KawPoW endpoint listens on `0.0.0.0:3333` (`--node.stratum-kawpow-addr`);
SHA-256 and Scrypt get 3335 and 3334, and an HTTP stats API 3336
(`--node.stratum-api-addr`). go-quai's own stratum README swaps the KawPoW and
SHA ports; the flag defaults above are what the node uses. go-quai's
stratum has no authentication: anyone who can reach the port can mine
through your node, so keep it on your network.

**rs-quai**: the same flags (`--node.stratum-enabled`, or `stratum-enabled =
true` under `[node]` in its config.toml), and optionally an auth token that
miners must send. The node doesn't make one; generate it:

    openssl rand -hex 16 > stratum-token

and give it to the node in the environment variable
`RSQ_STRATUM_AUTH_TOKEN` (or the `--rsq.stratum-auth-token` flag; it can't
go in config.toml). Without a token, rs-quai warns at start that anyone who
can reach the port can mine through it. Give the miner the same file with
`--auth-token-file` or `auth_token_file`; it sends it as `auth=<token>` and
never logs it. A miner with a missing or wrong token is refused as
`unauthorized`.

Either way: the node answers only addresses in its own zone; it raises any
share difficulty below 0.1 to 0.1; without `d=` its variable difficulty
starts at 50 and takes minutes to come down; and it drops a connection that
sends no share for 10 minutes. The stats API on 3336 has no authentication
either: bind it to 127.0.0.1 unless you need it elsewhere.

## Configuration

Every setting can be given as a flag or in a TOML file (`--config`); flags
win. The installed service reads `/etc/quai-kawpow-miner/miner.toml`.
`miner.example.toml` (in the release, or `dist/` in the source) has every
setting with a comment. A minimal file:

    api = "127.0.0.1:4073"            # for watch, the API and metrics

    [[pool]]
    url = "stratum+tcp://192.168.1.10:3333"
    user = "0xYOUR_ADDRESS.rig1"
    pass = "d=1"
    #auth_token_file = "/etc/quai-kawpow-miner/stratum-token"   # rs-quai token

`quai-kawpow-miner mine --check --config miner.toml` checks a file (pools,
address, token file, TLS, hooks, GPU settings) without touching a GPU: it
prints where the rewards go and `configuration OK`, or what is wrong.

### Settings

| Key | Flag | Default | Meaning |
|---|---|---|---|
| `devices` | `--devices 0,1,2` | every GPU | The GPUs to mine on (as `devices` lists them). |
| `api` | `--api ADDR` | off | The HTTP API, for `watch`, JSON and metrics. Keep it on 127.0.0.1 unless you mean to share it. |
| `lock` | `--lock 0..3` | 0 | The coinbase lockup (below). |
| `temp_limit` | `--temp-limit C` | 90 | A GPU this hot pauses until it cools to `temp_resume`; 0 turns it off. |
| `temp_resume` | `--temp-resume C` | limit − 15 | |
| `start_stagger` | `--start-stagger S` | 5 | Seconds between one GPU's DAG build and the next, so the rig's power draw rises in steps. |
| `log_file` | `--log-file PATH` | none | Also write the log here; it rotates at 50 MiB, keeping three old files. |
| `status_secs` | `--status-secs N` | 30 | Seconds between status lines (without the full screen). |
| `report_hashrate` | `--report-hashrate` | false | Send the hashrate to the pool every minute (`eth_submitHashrate`). |
| `dag_prefetch` | `--no-dag-prefetch` | true | Build the next epoch's DAG ahead when the card holds two. |
| `tui` | `--no-tui` | true | The full-screen view (only when stdout is a terminal). |
| `[[pool]]` | `--pool`, `--user`, `--pass`, `--auth-token-file` | | One table per pool, in order of preference (below). |
| `[tune]` | `--power-limit`, ... | | GPU settings ([GPU tuning](#gpu-tuning)). |
| `[hooks]` | `--hook-command`, `--hook-url` | | Alerts ([Monitoring and alerts](#monitoring-and-alerts)). |

### Pools

    [[pool]]
    url = "stratum+tcp://192.168.1.10:3333"   # or host:port; stratum+ssl:// for TLS
    user = "0xYOUR_ADDRESS.rig1"              # payout address . worker name
    pass = "d=1"                              # options, below
    auth_token_file = "/etc/quai-kawpow-miner/stratum-token"
    #tls_fingerprint = "AB:CD:..."            # TLS only; see TLS

Give more `[[pool]]` tables (or repeat `--pool`) for failover: after two
failed connections in a row the miner moves to the next pool, and after ten
minutes on a backup it checks the first one every minute and goes back as
soon as it answers. Every pool is checked when the miner starts.

The **worker name** (after the `.`) is how the node tells your rigs apart:
letters, digits, `-` and `_`, up to 64 characters. The URL may also carry
the login, as in kawpowminer and Quai's docs:
`stratum+tcp://0xADDRESS.rig1:d=1@host:3333`; `--user` and `--pass` win
over it.

**Password options**, separated by `,` (or `_`):

- `d=<difficulty>`: a fixed share difficulty. Aim for a share every 30 s or
  so: d ≈ your hashrate in MH/s ÷ 150.

  | Rig | d |
  |---|---|
  | one RTX 3060 Ti (~33 MH/s) | 0.2 |
  | ~150 MH/s | 1 |
  | seven RTX 3060 Ti (~230 MH/s) | 1.5 |

  The node raises anything below 0.1 to 0.1 and drops a connection without
  a share for 10 minutes. The miner warns when `d` is set so high that
  shares are too rare, or so low that its CPU check falls behind.
- `lock=`: set by `lock` / `--lock` instead (below).
- `frequency=<s>`: how often the node sends new work, 1 to 20 s (default 5).
- `auth=`: added by the miner from the token file. Don't write it here.

The miner refuses what the node would refuse or misread: a `.` between
options, a lock outside 0..3, a `d` above 1e6 (the node reads that as a
hash count), or a token containing `,` or `_`.

### Payout address, ledger and lockup

- **The address picks the ledger.** A Quai address (second byte below
  `0x80`) is paid in Quai; a Qi address (second byte `0x80` or above) in Qi.
- **The first byte is the zone** (`0x00` is cyprus1, `0x01` cyprus2, `0x10`
  paxos1, ...). The node accepts only addresses in its own zone.
- The miner checks the address (40 hex digits) before connecting, because a
  node would quietly pad or cut a short or long one and pay someone else.
- **`lock` sets the coinbase lockup.** A longer lockup earns a reward bonus
  that shrinks over the years. It applies to both ledgers:

  | `lock` | Locked for | Bonus (2026) |
  |---|---|---|
  | 0 | the node's default (normally 2 weeks) | none |
  | 1 | 3 months | about ×1.03 |
  | 2 | 6 months | about ×1.085 |
  | 3 | 1 year | about ×1.21 |

  Solo mining over RPC can't set it; that is one reason this miner uses the
  node's stratum.
- **Accepted shares** count everything the node accepted: shares at your
  `d` that show the miner is working, as well as workshares and blocks. The
  node answers them all the same way; its stats API (port 3336) tells them
  apart.

## Running in a terminal

    quai-kawpow-miner mine --config miner.toml

uses every GPU through one connection and one worker, and takes over the
terminal:

- a header with the worker, where it pays (`Qi · cyprus1 · lock 0`), the
  pool and the uptime;
- the rig's hashrate and its last ten minutes as a graph; shares (accepted,
  rejected, stale, lost, and the age of the last one), the job, epoch and
  difficulty, the pool's answer time, the rig's power and efficiency;
- a row per GPU: state, hashrate, shares, temperature, fan, power against
  its limit, kH/W, clocks, utilization, VRAM, P-state, PCIe link, and why its
  clocks are held down (power cap, thermal ...). Temperatures turn yellow at
  70 °C and red at 80 °C; columns drop by priority on narrow terminals;
- the log.

`q` twice (or Ctrl-C) stops the miner; arrows, PgUp/PgDn and Home/End scroll
the log. Closing the terminal stops it too. `--no-tui`, or output that is
not a terminal, prints log lines and a status line instead.

## Running as a service

`install.sh` installs the miner as a systemd service that starts at boot and
restarts after a failure:

    sudo ./install.sh --config miner.toml --token stratum-token --start

(from the release directory; in the source, `dist/install.sh` after `cargo
build --release`). It:

- puts the binary in `/usr/local/bin/quai-kawpow-miner`, your config in
  `/etc/quai-kawpow-miner/miner.toml`, and the token (`--token`) in
  `/etc/quai-kawpow-miner/stratum-token`, readable only by root and the
  miner's user;
- creates the user the miner runs as (`miner`, or `--user NAME`) if it
  doesn't exist;
- checks the config as that user (`mine --check`, so an unreadable token
  shows up now, not at boot), and enables the service only when it passes;
  otherwise it says what's wrong and exits 2;
- starts it with `--start`.

Without `--config`, a first install puts the example in place, which fails
the check until you edit it.

**Upgrading:** run `install.sh` from the new release. It stops the miner,
replaces the binary, keeps your config (unless you pass `--config`; the old
one is kept as `.bak`), and starts it again.

**Day to day:**

    quai-kawpow-miner watch --config /etc/quai-kawpow-miner/miner.toml   # the full screen
    systemctl status quai-kawpow-miner
    journalctl -fu quai-kawpow-miner                                     # the log
    sudo systemctl restart quai-kawpow-miner                             # after editing the config
    sudo systemctl stop quai-kawpow-miner                                # until the next boot

The service sets GPU persistence mode, applies `[tune]` as root, then mines
as its user. `systemctl stop` stops it cleanly (it finishes the GPUs and
sends the last alerts). A crash or a run-time failure restarts it after
10 s, for as long as it takes. A problem a restart can't fix (a bad config,
or every pool refusing the login) leaves it stopped, with the reason in the
log.

**Uninstalling:**

    sudo /usr/local/share/quai-kawpow-miner/uninstall.sh

stops and disables the service, puts the GPUs' power limits, clocks and fans
back to their defaults, and removes the service and the binary. It keeps
the config, the token and the logs unless you add `--purge` (it asks first;
`--yes` skips the question). `--remove-user` also deletes the user, but only
if `install.sh` created it.

**Exit codes**, for any supervisor:

| Code | Meaning | Restart? |
|---|---|---|
| 0 | stopped on purpose (Ctrl-C, `q`, SIGTERM) | no |
| 1 | a run-time failure: every GPU failed, a GPU stopped hashing, the stratum client died | yes |
| 2 | needs a fix: bad configuration, missing NVIDIA libraries, or every pool refused the login | no |

## Monitoring and alerts

### watch

`quai-kawpow-miner watch` shows the full screen of a miner running
elsewhere: as a service, in another session, or on another machine whose
API you can reach. It reads the API once a second (`--config FILE` takes the
address from its `api`, `--api HOST:PORT` gives it directly, and the default
is 127.0.0.1:4073). `q` closes it; the miner keeps running. If the miner
restarts, the bottom line says so until it answers again.

### API and metrics

With `api` set (e.g. `127.0.0.1:4073`):

- `GET /`: everything as JSON: hashrate, shares, pool, job, payout, and
  per GPU its state, hashrate, shares, temperature, power, clocks, fan,
  utilization, VRAM, P-state, throttle reasons and PCIe link;
- `GET /metrics`: Prometheus (`qkm_hashrate_hashes_per_second`,
  `qkm_shares_total{result=...}`, `qkm_gpu_temperature_celsius{device,bus,name}`
  and more);
- `GET /health`: `ok`;
- `GET /view`: what the full screen shows, with the last 1000 log lines
  (what `watch` reads).

Requests are capped at 8 KiB and 2 s each, and there is no CORS header. The
API has no password: bind it to a public address only on a network you
trust.

### Alerts (event hooks)

The miner can tell you when something needs attention, by running a
program, posting to a webhook (Discord, Slack, ntfy, or your own), or both:

    [hooks]
    webhook = "https://ntfy.sh/your-topic"     # or a Discord / Slack webhook URL
    format = "text"                            # plain text (ntfy); default json
    #command = "/usr/local/bin/on-miner-event"  # an absolute path
    hashrate_low_mhs = 180                     # alert below this...
    hashrate_low_mins = 5                      # ...for this long
    #events = ["default", "share_rejected"]
    #pool_down_mins = 2
    #interval_secs = 60
    #timeout_secs = 10

`--hook-url` and `--hook-command` set the first two from the command line.
`mine --check` validates the section.

| Event | When | Sent by default |
|---|---|---|
| `miner_stopping` | the miner stops: on request, or why it must | yes |
| `pool_failover` / `pool_primary_back` | to a backup pool, and back to the first | yes |
| `pool_down` / `pool_up` | no usable pool for `pool_down_mins`, and usable again | yes |
| `auth_refused` | a pool refused the address, lock or token | yes |
| `gpu_reset` / `gpu_failed` | a GPU reset after errors; a GPU given up | yes |
| `gpu_too_hot` | a GPU reached `temp_limit` and paused | yes |
| `hashrate_low` / `hashrate_ok` | the rig below `hashrate_low_mhs` for `hashrate_low_mins`, and back | yes, with a threshold |
| `miner_started`, `pool_connected`, `pool_disconnected`, `share_rejected`, `gpu_error`, `gpu_cooled` | the routine | no |

`events` picks them: names, `all`, or `default` plus more. `share_rejected`
is a refusal by the pool; stale shares and shares lost with a connection
are covered by the pool events. No alert can say "block found": the node
answers a block and a workshare the same way.

A webhook gets a JSON object (`event`, `message`, `detail`, `gpu`, `gpus`,
`worker`, `hashrate_mhs`, `folded`, `time`) with the one-line message also
in `content` and `text`, which is what Discord and Slack show. With
`format = "text"` it gets just the message, e.g. `rig1: gpu3: 91C reached
the 90C limit; paused until 75C`. A program gets the event name as `$1`
and the same fields in `QKM_EVENT`, `QKM_MESSAGE`, `QKM_DETAIL`, `QKM_GPU`,
`QKM_GPUS`, `QKM_WORKER`, `QKM_HASHRATE_MHS`, `QKM_FOLDED` and `QKM_TIME`.
It runs without a shell, with its output discarded, and is killed after
`timeout_secs`.

Alerts never slow mining: they queue for one delivery thread. A burst of
one event is folded: the first goes out at once, and the latest of the rest
after `interval_secs`, as `... (and 12 more like it, on gpu0 gpu2 gpu5)`.
The auth token is never passed on, the webhook URL (which often holds a
secret) is logged only by its host, and a failed delivery is logged without
stopping anything.

## TLS

Quai nodes serve stratum unencrypted, auth token included. To mine across
a network you don't trust, put a TLS proxy (stunnel, nginx `stream`,
HAProxy) in front of the node and point the miner at it with
`stratum+ssl://` (or `stratum+tls://`). With stunnel on the node:

    # A self-signed certificate, and its fingerprint for the miner.
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
        -days 3650 -subj /CN=quai-stratum -keyout stratum.key -out stratum.crt
    cat stratum.crt stratum.key > /etc/stunnel/stratum.pem
    openssl x509 -noout -fingerprint -sha256 -in stratum.crt

    # /etc/stunnel/quai-stratum.conf
    [quai-stratum]
    accept = 0.0.0.0:3443
    connect = 127.0.0.1:3333
    cert = /etc/stunnel/stratum.pem

Then on the miner:

    [[pool]]
    url = "stratum+ssl://node.example:3443"
    user = "0xYOUR_ADDRESS.rig1"
    pass = "d=1"
    tls_fingerprint = "AB:CD:..."       # the fingerprint printed above

and keep the node's stratum port off the network (bind it to 127.0.0.1 or
firewall it), so nothing can reach it unencrypted.

The pool's certificate is checked one of three ways:

- **`tls_fingerprint`** (`--tls-fingerprint`): only the certificate with
  this SHA-256, whatever its name, issuer or dates. Use it for a self-signed
  certificate. Colons and case don't matter, and openssl's `sha256
  Fingerprint=...` line works as it is.
- **`tls_ca`** (`--tls-ca`): a certificate for the pool's host name, issued
  by a CA in this PEM file (your own CA).
- **neither**: a certificate for the host name from a public CA (Mozilla's
  list, built in).

When a certificate is refused, the log says why and prints its fingerprint,
so you can compare it with your proxy's. TLS 1.2 and 1.3 are supported
through rustls; OpenSSL isn't needed.

## GPU tuning

    sudo quai-kawpow-miner tune --power-limit 200          # every GPU
    sudo quai-kawpow-miner tune --config miner.toml        # the file's [tune]
    quai-kawpow-miner tune --show                          # the current settings
    sudo quai-kawpow-miner tune --reset                    # back to the defaults

In the config:

    [tune]                 # every GPU
    power_limit = 200      # W
    #lock_core_clocks = [1400, 1800]   # MHz
    #lock_mem_clocks = [5000, 9251]    # MHz
    #core_offset = 0                   # MHz
    #mem_offset = 0                    # MHz
    #fan = 70                          # %, turns automatic fan control off

    [tune.gpu.0]           # one GPU's own settings
    power_limit = 150

Settings need root and last until `tune --reset`, a reboot or a driver
reload. They are matched to GPUs by PCI bus id, because CUDA and NVML may
number GPUs differently. The service applies `[tune]` at every start. `mine`
applies it when run as root; otherwise it checks the power limits are as
configured and says so in one line.

**The power limit matters most.** KawPoW on an RTX 3060 Ti is bound by power
and memory bandwidth. Per card (Quai mainnet, epoch 610):

| Power limit | MH/s | kH/W |
|---|---|---|
| 150 W | 21.3 | 143 |
| 175 W | 27.5 | 157 |
| 200 W | 32.1 | **161** (the most efficient) |
| 225 W | 34.1 | 152 |

Memory clock offsets and locked clocks did not help on these cards: under
CUDA the memory runs at 9251 MHz, and a faster memory clock takes power the
cores then lack.

To let an operator tune without a password, allow only that command in
sudoers (`visudo -f /etc/sudoers.d/qkm-tune`):

    miner ALL=(root) NOPASSWD: /usr/local/bin/quai-kawpow-miner tune *

## Octominer rigs

Octominer's fan board has a hardware watchdog. Without OctOS or HiveOS
nothing feeds it, and it power-cycles the rig 25 minutes after every boot.
Install [octofan-watchdog](https://github.com/mpoletiek/octofan-watchdog)
before you start mining:

    curl -LO https://github.com/mpoletiek/octofan-watchdog/releases/latest/download/octofan-watchdog-x86_64-linux.tar.gz
    tar -xzf octofan-watchdog-x86_64-linux.tar.gz
    sudo ./octofan-watchdog-x86_64-linux/install.sh

## Troubleshooting

Start with `quai-kawpow-miner mine --check --config miner.toml`, and for
the service, `journalctl -u quai-kawpow-miner -n 50`.

| You see | Why, and what to do |
|---|---|
| `NVRTC (libnvrtc.so) is not installed or not on the library path` | Install NVRTC (`cuda-nvrtc-13-1` on Ubuntu 26.04) and add its directory to `/etc/ld.so.conf.d/` ([Quick start](#quick-start), step 1), then `sudo ldconfig`. |
| `the NVIDIA driver library (libcuda.so.1) is not installed` | Install the NVIDIA driver and reboot; `nvidia-smi` must work first. |
| `epoch N needs a 5.77 GiB DAG but only 5.36 GiB of GPU memory is free` | The card is too small, or a desktop or another program is using its memory. Close it, or leave that GPU out with `devices`. |
| `authorization refused: unauthorized` | rs-quai wants an auth token: set `auth_token_file` to a file holding the node's `RSQ_STRATUM_AUTH_TOKEN`. |
| `authorization refused: address is not internal to this zone` | The address's first byte is a different zone from the node's: use an address in the node's zone, or a node in the address's zone. |
| `the payout address ... has N hex digits, not 40` | Copy the whole address: `0x` and 40 hex digits. |
| `every pool refused authorization ...; check the address, lock and auth token` | Every pool turned the login down; the reasons come before it in the log. The miner stops (exit 2) rather than retry forever. |
| `disconnected: ...: connect ...: Connection refused` (repeating) | The node's stratum is off, on another port, or firewalled ([Setting up the node](#setting-up-the-node)). The miner keeps retrying. |
| `difficulty d: one share every N min ...` | `d` is too high for the hashrate: the node drops a connection without a share for 10 minutes. Lower `d`. |
| `N GPU solutions dropped: the CPU check cannot keep up; raise the share difficulty (d=)` | `d` is too low: the CPU can't verify shares as fast as the GPUs find them. Raise `d`. |
| A GPU shows `too hot` and pauses | It reached `temp_limit`. Check its fan and airflow, or lower its power limit (`[tune.gpu.N]`). |
| One GPU's hashrate well below the others' | Usually heat: look at its temperature and the Throttle column (`thermal`). |
| `tuning: not root and the GPUs are not as configured` | `mine` without root can't apply `[tune]`: run `sudo quai-kawpow-miner tune --config ...` (the service does this itself). |
| `install.sh`: `the config does not check out` | The check's message comes above it. Fix the config, then run `install.sh` again. |
| `watch`: `cannot read the miner's screen` | The miner isn't running, or has no `api` set, or it's at another address (`--api HOST:PORT`). |
| The rig reboots every 25 minutes | An Octominer without its fan watchdog: install [octofan-watchdog](#octominer-rigs). |
| A few stale or lost shares | Normal: stale ones were for work the node had replaced; lost ones were in flight when a connection dropped. |

## Command reference

| Command | What it does |
|---|---|
| `mine` | Mine (options below). |
| `watch [--config FILE \| --api HOST:PORT]` | The full screen of a running miner, from its API. |
| `devices` | The GPUs CUDA and NVML see: index, PCI bus, name, memory, temperature, power limit. |
| `tune [--config FILE] [settings] [--show \| --reset] [--devices ...]` | Apply, show or reset GPU settings; needs root to change them. |
| `verify [--heights 1,4577070]` | Compare the GPU with the CPU reference, hash by hash, and check the search kernel misses nothing. |
| `bench` | Measure the hashrate (and sweep kernel launch settings). |
| `compile`, `kernel` | Compile, or print, the generated CUDA kernel for a height. |

`mine` options (each has a config key; see [Settings](#settings)):

- Pools: `--pool URL` (repeat for failover), `--user 0xADDRESS.worker`,
  `--pass OPTIONS`, `--auth-token-file FILE`, `--lock 0..3`,
  `--tls-fingerprint SHA256`, `--tls-ca PEM`.
- GPUs: `--devices 0,1,2` (or `--device N` for one), `--temp-limit C`,
  `--temp-resume C`, `--start-stagger SECS`, `--no-dag-prefetch`.
- GPU settings: `--power-limit W`, `--lock-core-clocks MIN,MAX`,
  `--lock-mem-clocks MIN,MAX`, `--core-offset MHZ`, `--mem-offset MHZ`,
  `--fan PCT`.
- Output: `--no-tui`, `--log-file PATH`, `--status-secs N`, `--api ADDR`,
  `--report-hashrate`.
- Alerts: `--hook-command PATH`, `--hook-url URL`.
- `--config FILE`, `--check`.

`quai-kawpow-miner <command> --help` lists them all with their defaults.

## Building from source

You need Rust 1.97 or later and a C compiler. The CUDA toolkit isn't needed
to build: the driver and NVRTC are loaded when the miner runs.

    git clone https://github.com/mpoletiek/quai-kawpow-miner
    cd quai-kawpow-miner
    cargo build --release        # target/release/quai-kawpow-miner
    sudo dist/install.sh --binary target/release/quai-kawpow-miner --config miner.toml --start

The release build is linked against glibc 2.28 with
[cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild)
(`cargo zigbuild --release --target x86_64-unknown-linux-gnu.2.28`); a plain
`cargo build` needs the build machine's glibc or newer.

## How it works

Quai's KawPoW work is merge-mined with Ravencoin: the miner hashes a
Ravencoin-style donor header that commits to a Quai block, handed out by
the node's stratum.

- **The node's own algorithm.** KawPoW comes from `crates/qkm-kawpow`, ported
  from rs-quai's `rsq-pow`, which is verified against go-quai across the whole
  mainnet chain; it passes go-quai's Ravencoin block vectors. The GPU
  kernels are generated from the same code and compiled for your GPU when
  the miner starts (NVRTC). Every share is checked on the CPU before it is
  sent, so the miner never submits a bad share.
- **Every GPU, one process.** One stratum connection and one worker for the
  rig; each GPU searches its own part of the nonce space; the light cache
  and the compiled kernels are shared.
- **No duplicate shares.** Quai nodes re-send the same work under a new job
  id after every accepted share. This miner starts every job at a random
  nonce and never sends the same share twice, so the node never refuses its
  shares as duplicates (some miners lose 11% this way; see Benchmarks).

What it does on its own:

- **Pools.** Reconnects with backoff, fails over after two failures in a
  row, and returns to the first pool once it answers. Reconnects when the
  pool stops answering shares (three in a row), sends no job for three
  minutes, or doesn't answer the login in 10 s. Nothing a pool sends can
  crash it: an over-long line, bad hex or an impossible height is refused,
  and pool text is stripped of control characters before it is logged. A
  refusal no retry can fix (wrong zone, lock or token) stops the miner with
  the reason.
- **Epochs.** Near the end of an epoch the next light cache and kernel are
  prepared ahead, and the next DAG too when the card holds two. A job from
  an unexpected epoch (a stale template) is followed only when a second job
  confirms it.
- **GPUs.** CUDA errors are retried after a pause; after three in a row the
  device is reset (an illegal address or ECC fault ruins its context);
  after eight the GPU is given up. A GPU at the temperature limit pauses
  until it cools. A GPU that stops hashing for two minutes (a hung kernel
  or driver) ends the miner with code 1, for the service to restart it.
- **Shares.** A GPU result that doesn't match the CPU's check is logged as a
  GPU bug and never sent. The CPU check queue is bounded; drops are counted
  and logged with a hint to raise `d`.

## Benchmarks

Measured 2026-10-09 on an Octominer X12 Ultra, six RTX 3060 Ti at a 200 W
power limit each (clocks unlocked, driver 595.99, epoch 609), mining Quai
mainnet through rs-quai's stratum at the node's minimum difficulty (0.1),
one miner at a time, 15 minutes each after 90 s of warm-up.

| Miner | Version | Hashrate (own report) | Refused by the node | Dev fee | Effective |
|---|---|---|---|---|---|
| **quai-kawpow-miner** | 0.1.0 | 186.0 MH/s (two runs: 186.0, 185.7) | 0 | 0% | **186.0 MH/s** |
| Rigel | 1.23.2 | 187.0 MH/s | 0 | 1% | 185.1 MH/s |
| SRBMiner-MULTI | 3.7.3 | 181.7 MH/s | 0 | 0.85% | 180.2 MH/s |
| T-Rex | 0.26.8 | 190.7 MH/s | 11% (duplicate shares) | 1% | 168 MH/s |
| kawpowminer | 1.2.4 | 187.4 MH/s | 11% (duplicate shares) | 0% | 166 MH/s |

- Power-bound at the same 200 W (1,199 W for the six), the kernels are
  within 2.5% of each other by their own reports; quai-kawpow-miner ties
  Rigel and is 2.3% ahead of SRBMiner.
- One card against T-Rex's benchmark mode (2026-10-11, GPU 1 alone, epoch
  610, two interleaved rounds each): at 200 W they tie (T-Rex 31.91 MH/s,
  quai-kawpow-miner 31.93); at 225 W T-Rex reports 1.4% more (34.30 against
  33.84 MH/s). That is T-Rex's own figure: it refuses to run under a
  profiler, so its work can't be counted independently.
- T-Rex and kawpowminer lose 11% of their shares on a Quai node: the node
  re-sends the same work under a new job id after every accepted share,
  and they search it again from the same nonce, so the node refuses what
  they find again as duplicates.
- The node's own count of valid shares agrees within its statistical error
  (about 5% at this share rate): 206 ± 10 and 194 ± 10 MH/s for this
  miner's two runs, 193 ± 10 for Rigel, 175 ± 9 for SRBMiner.
- The kernel is at the memory's limit: at 225 W it reads 93-94% of the
  DAG bandwidth the card's CUDA memory clock allows, and at 200 W the power
  cap holds it at 90%. Launch settings, occupancy, unrolling, the DAG load
  path and memory clocks change the hashrate by at most 1-2%, and none
  wins (what was measured is listed in `crates/qkm-core/src/kernel.rs`).

## Correctness and tests

- `qkm-kawpow` passes go-quai's KawPoW tests: the Ravencoin block vectors,
  the first KawPoW block and a GPU-mined share
  (`cargo test -p qkm-kawpow --release -- --ignored`).
- `verify` compares the GPU with `qkm-kawpow` hash by hash: 0 differences in
  768 mix digests and final hashes at epochs 0, 300 and 608 (plus the next
  period's kernel at each); over 2^20 nonces the search kernel reports
  exactly the nonces whose hash meets the target; over 2^31 nonces at a
  1/2^24 target it finds the expected number of solutions (128 ± 11).
- On mainnet: 0 rejected shares across runs of 7 GPUs from one process,
  through rs-quai's stratum, over plain TCP and over TLS.
- `cargo test` runs the protocol, client, TLS, mock-pool, configuration,
  alert, API and screen tests on the CPU, including hostile pool input. The
  GPU end-to-end tests (mining through a mock pool that checks shares the
  way rs-quai does, across epoch changes, failover and alerts, on every GPU)
  need the GPUs to themselves:

      cargo test --release -p quai-kawpow-miner --test gpu_mock -- --ignored --test-threads 1

Layout: `crates/qkm-kawpow` (KawPoW on the CPU: the reference), `crates/qkm-core`
(the program, kernel sources, epochs, CPU verification), `crates/qkm-cuda`
(the CUDA backend), `crates/qkm-stratum` (the stratum client, TLS and a mock
pool), `crates/quai-kawpow-miner` (the miner, its screen, API, alerts,
tuning and the binary); `dist/` (the service, `install.sh`, `uninstall.sh`,
the example config).

## Roadmap

- Solo mining over the node's RPC (`quai_getBlockTemplate` /
  `quai_submitKawpowBlock`).
- A HiveOS package.
- AMD GPUs (an OpenCL or HIP backend).

## License

GPL-3.0-only. `crates/qkm-kawpow` is ported from rs-quai's GPL-3.0
`rsq-pow`.
