#!/bin/sh
# Installs quai-kawpow-miner as a systemd service that starts at boot.
#
#   sudo dist/install.sh [--binary PATH] [--config PATH] [--token PATH]
#                        [--user NAME] [--start]
#
#   --binary  the miner (default: target/release/quai-kawpow-miner, or
#             quai-kawpow-miner next to this script)
#   --config  your miner.toml (default: keep the installed one; on a first
#             install, the example, which you then edit)
#   --token   the node's stratum token file, installed readable only by
#             root and the miner's user; point auth_token_file at
#             /etc/quai-kawpow-miner/stratum-token
#   --user    the user the miner runs as (default: miner; created as a
#             system user if missing)
#   --start   start (or restart) it now, not only at the next boot
#
# Run it again to upgrade: an installed config is only replaced by --config.
set -eu

die() { echo "install.sh: $*" >&2; exit 1; }

here=$(cd "$(dirname "$0")" && pwd)
binary=""
config=""
token=""
user=miner
start=no
while [ $# -gt 0 ]; do
    case $1 in
        --binary) binary=${2:?--binary needs a path}; shift 2 ;;
        --config) config=${2:?--config needs a path}; shift 2 ;;
        --token) token=${2:?--token needs a path}; shift 2 ;;
        --user) user=${2:?--user needs a name}; shift 2 ;;
        --start) start=yes; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "run it as root (sudo)"
command -v systemctl >/dev/null || die "systemd is needed"
case $user in
    *[!a-z0-9_-]*|"") die "--user: letters, digits, _ and - only" ;;
esac

if [ -z "$binary" ]; then
    for b in "$here/../target/release/quai-kawpow-miner" "$here/quai-kawpow-miner"; do
        if [ -x "$b" ]; then binary=$b; break; fi
    done
    [ -n "$binary" ] || die "no miner binary found; build it (cargo build --release) or give --binary"
fi
[ -x "$binary" ] || die "$binary is not an executable"
for f in "$here/quai-kawpow-miner.service" "$here/miner.example.toml"; do
    [ -f "$f" ] || die "$f is missing (run the script from the dist directory it came in)"
done

etc=/etc/quai-kawpow-miner
unit=/etc/systemd/system/quai-kawpow-miner.service

# The user: a system account with a home under /var/lib (CUDA keeps a cache
# there), in the video group where the distribution has one.
if ! id "$user" >/dev/null 2>&1; then
    groups=""
    getent group video >/dev/null && groups="--groups video"
    # shellcheck disable=SC2086
    useradd --system --create-home --home-dir "/var/lib/$user" \
        --shell /usr/sbin/nologin $groups "$user"
    echo "created the system user $user"
fi

# Stop a running miner before replacing its binary.
was_active=no
if systemctl is-active --quiet quai-kawpow-miner; then
    was_active=yes
    systemctl stop quai-kawpow-miner
fi

install -m 0755 "$binary" /usr/local/bin/quai-kawpow-miner
echo "installed $(/usr/local/bin/quai-kawpow-miner --version 2>/dev/null || echo the miner) at /usr/local/bin/quai-kawpow-miner"

install -d -m 0755 "$etc"
if [ -n "$config" ]; then
    [ -f "$config" ] || die "no config at $config"
    if [ -f "$etc/miner.toml" ] && ! cmp -s "$config" "$etc/miner.toml"; then
        cp "$etc/miner.toml" "$etc/miner.toml.bak"
        echo "kept the old config as $etc/miner.toml.bak"
    fi
    install -m 0644 "$config" "$etc/miner.toml"
    echo "installed $etc/miner.toml"
elif [ ! -f "$etc/miner.toml" ]; then
    install -m 0644 "$here/miner.example.toml" "$etc/miner.toml"
    first_config=yes
    echo "installed the example config at $etc/miner.toml: edit it (pool, address) before starting"
fi

if [ -n "$token" ]; then
    [ -f "$token" ] || die "no token file at $token"
    install -m 0640 -o root -g "$(id -gn "$user")" "$token" "$etc/stratum-token"
    echo "installed the token at $etc/stratum-token (readable by root and $user)"
fi

sed "s/^User=.*/User=$user/" "$here/quai-kawpow-miner.service" > "$unit.tmp"
chmod 0644 "$unit.tmp"
mv "$unit.tmp" "$unit"
systemctl daemon-reload
systemctl enable quai-kawpow-miner >/dev/null
echo "installed $unit; enabled at boot"

if [ "${first_config:-no}" = yes ]; then
    echo "not started: edit $etc/miner.toml, then: sudo systemctl start quai-kawpow-miner"
elif [ $start = yes ] || [ $was_active = yes ]; then
    systemctl start quai-kawpow-miner
    sleep 3
    if systemctl is-active --quiet quai-kawpow-miner; then
        echo "running. Follow it: journalctl -fu quai-kawpow-miner"
    else
        echo "it did not stay up:" >&2
        journalctl -u quai-kawpow-miner -n 20 --no-pager >&2
        exit 1
    fi
else
    echo "it starts at the next boot, or now with: sudo systemctl start quai-kawpow-miner"
fi
