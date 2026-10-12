# Security

## Reporting a problem

Please report security problems privately, not in a public issue: use
GitHub's private reporting, under this repository's **Security** tab,
**Report a vulnerability**
(<https://github.com/mpoletiek/quai-kawpow-miner/security/advisories/new>).

Say what you found, how to reproduce it, and what it lets an attacker do.
You will get a reply as soon as possible, and credit in the fix's release
notes if you want it.

## What counts

The miner runs unattended on mining rigs, often as a system service, and
talks to pools and webhooks over the network. Problems worth reporting
privately include:

- anything that leaks the node's stratum auth token or a webhook URL (into
  the log, the API, alert payloads, hook programs or files with loose
  permissions);
- input from a pool or a TLS server that crashes the miner, makes it hang,
  grow without bound, or run code;
- a way to make the miner submit shares, or pay rewards, to an address other
  than the configured one;
- the HTTP API, `watch`, or event hooks being used to reach or run
  something they shouldn't;
- `install.sh` or `uninstall.sh` doing more than they say as root (for
  example, through a crafted config or user name).

Bugs without a security impact (a wrong number on the screen, a mining
error) are welcome as ordinary issues.

## Supported versions

Fixes go into the next release. Please check the latest release before
reporting.
