#!/bin/sh
# Removes the quai-kawpow-miner service that install.sh set up.
#
#   sudo /usr/local/share/quai-kawpow-miner/uninstall.sh [--purge] [--remove-user] [--yes]
#   (or dist/uninstall.sh from the source)
#
# Always: stops and disables the service, puts the GPUs' power limits,
# clocks and fans back to their defaults, and removes the unit, the binary
# and /usr/local/share/quai-kawpow-miner.
#
#   --purge        also delete /etc/quai-kawpow-miner (the config and the
#                  stratum token) and /var/log/quai-kawpow-miner; asks first
#                  unless --yes
#   --remove-user  also delete the user, only when install.sh created it
#                  (an account that existed before is never touched)
#   --yes          do not ask
set -eu

die() { echo "uninstall.sh: $*" >&2; exit 1; }

purge=no
remove_user=no
yes=no
while [ $# -gt 0 ]; do
    case $1 in
        --purge) purge=yes; shift ;;
        --remove-user) remove_user=yes; shift ;;
        --yes|-y) yes=yes; shift ;;
        -h|--help) sed -n '2,/^set -eu/p' "$0" | grep '^#'; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

[ "$(id -u)" -eq 0 ] || die "run it as root (sudo)"

bin=/usr/local/bin/quai-kawpow-miner
etc=/etc/quai-kawpow-miner
share=/usr/local/share/quai-kawpow-miner
logs=/var/log/quai-kawpow-miner
unit=/etc/systemd/system/quai-kawpow-miner.service

if [ $purge = yes ] && [ $yes = no ]; then
    if [ -t 0 ]; then
        printf 'Delete %s (the config and the stratum token) and %s? [y/N] ' "$etc" "$logs"
        read -r answer
        case $answer in
            y|Y|yes|YES) ;;
            *) echo "nothing removed"; exit 1 ;;
        esac
    else
        die "--purge deletes the config and the token: add --yes to confirm"
    fi
fi

created_user=""
[ -f "$share/created-user" ] && created_user=$(cat "$share/created-user")

if command -v systemctl >/dev/null; then
    if systemctl is-active --quiet quai-kawpow-miner 2>/dev/null; then
        systemctl stop quai-kawpow-miner
        echo "stopped the miner"
    fi
    systemctl disable quai-kawpow-miner >/dev/null 2>&1 || true
fi

# The GPU settings the service applied last until a reboot: undo them now.
if [ -x "$bin" ]; then
    if "$bin" tune --reset >/dev/null 2>&1; then
        echo "put the GPUs' power limits, clocks and fans back to their defaults"
    else
        echo "could not reset the GPU settings (they return to their defaults at the next reboot)"
    fi
fi

if [ -f "$unit" ]; then
    rm -f "$unit"
    echo "removed $unit"
fi
if command -v systemctl >/dev/null; then
    systemctl daemon-reload
    systemctl reset-failed quai-kawpow-miner >/dev/null 2>&1 || true
fi
if [ -e "$bin" ]; then
    rm -f "$bin"
    echo "removed $bin"
fi

if [ $purge = yes ]; then
    rm -rf "$etc" "$logs"
    echo "deleted $etc and $logs"
else
    [ -d "$etc" ] && echo "kept $etc (the config and the token; --purge deletes them)"
    [ -d "$logs" ] && echo "kept $logs (--purge deletes it)"
fi

if [ $remove_user = yes ]; then
    if [ -n "$created_user" ] && id "$created_user" >/dev/null 2>&1; then
        userdel --remove "$created_user" 2>/dev/null || userdel "$created_user"
        echo "deleted the user $created_user"
    else
        echo "kept the user: install.sh did not create it"
    fi
elif [ -n "$created_user" ]; then
    echo "kept the user $created_user (--remove-user deletes it)"
fi

# Last, as this script may be running from there.
rm -rf "$share"
echo "quai-kawpow-miner is uninstalled"
