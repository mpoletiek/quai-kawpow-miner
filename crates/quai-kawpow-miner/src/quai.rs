//! Quai specifics a miner must get right before it connects: the payout
//! address (which ledger, Quai or Qi, and which zone it pays in), the
//! coinbase lockup (`lock=`), and the stratum password.
//!
//! Rules from go-quai (`common/address.go`, `params/protocol_params.go`)
//! and rs-quai's stratum (a port of go-quai's):
//! - an address is 20 bytes; its first byte is the zone,
//!   `region << 4 | zone` (`0x00` is cyprus1), and the node refuses an
//!   address from another zone ("address is not internal to this zone").
//!   The node does not check the length, so a short or long address whose
//!   first byte matches would be padded or cut and paid elsewhere: we check;
//! - the second byte's top bit picks the ledger: set for Qi, clear for Quai;
//! - `lock=0..3` picks the coinbase lockup: 2 weeks, 3 months, 6 months or
//!   a year, with a reward multiplier for 1..3 that shrinks over the years
//!   (about x1.03, x1.085 and x1.21 in 2026). 0 leaves the node's own
//!   default (`--node.coinbase-lockup`, normally 0);
//! - password options are separated by `,` or `_`, so a token holding either
//!   can never match.

use std::fmt;

/// The ledger an address pays in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ledger {
    /// The Quai (account) ledger.
    Quai,
    /// The Qi (UTXO) ledger.
    Qi,
}

impl fmt::Display for Ledger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Ledger::Quai => "Quai",
            Ledger::Qi => "Qi",
        })
    }
}

/// A checked payout address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Payout {
    /// The 20 bytes.
    pub bytes: [u8; 20],
    /// Its ledger.
    pub ledger: Ledger,
}

impl Payout {
    /// The zone's name, e.g. `cyprus1` (region and zone 0..2), or
    /// `region R zone Z` for other values.
    pub fn zone(&self) -> String {
        zone_name(self.bytes[0])
    }
}

/// `cyprus1` for 0x00, `paxos2` for 0x11, ...
pub fn zone_name(first: u8) -> String {
    let (region, zone) = (first >> 4, first & 0x0f);
    const REGIONS: [&str; 3] = ["cyprus", "paxos", "hydra"];
    match REGIONS.get(region as usize) {
        Some(r) if zone < 3 => format!("{r}{}", zone + 1),
        _ => format!("region {region} zone {zone}"),
    }
}

/// Parses the payout address at the start of a stratum user
/// (`0xADDRESS.worker`): exactly 40 hex digits after `0x`.
pub fn parse_payout(user: &str) -> Result<Payout, String> {
    let address = user.split('.').next().unwrap_or("");
    let hex = address
        .strip_prefix("0x")
        .or_else(|| address.strip_prefix("0X"))
        .ok_or_else(|| format!("the payout address {address:?} must start with 0x"))?;
    if hex.len() != 40 {
        return Err(format!(
            "the payout address {address:?} has {} hex digits, not 40",
            hex.len()
        ));
    }
    let b = qkm_stratum::proto::unhex(hex)
        .map_err(|_| format!("the payout address {address:?} is not hex"))?;
    let mut bytes = [0u8; 20];
    bytes.copy_from_slice(&b);
    let ledger = if bytes[1] & 0x80 != 0 {
        Ledger::Qi
    } else {
        Ledger::Quai
    };
    Ok(Payout { bytes, ledger })
}

/// Checks the worker name after the address (`0xADDRESS.worker`); the
/// node keeps only the part up to the next `.`.
pub fn check_worker(user: &str) -> Result<(), String> {
    let mut parts = user.split('.');
    parts.next();
    let Some(w) = parts.next() else {
        return Ok(());
    };
    if parts.next().is_some() {
        return Err(format!(
            "the user {user:?} has more than one '.'; the node would drop what follows the second"
        ));
    }
    if w.is_empty() || w.len() > 64 {
        return Err(format!("the worker name {w:?} must be 1 to 64 characters"));
    }
    if !w
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!(
            "the worker name {w:?} may hold only letters, digits, '-' and '_'"
        ));
    }
    Ok(())
}

/// The coinbase lockup a `lock` value asks for.
pub fn lock_meaning(lock: u8) -> &'static str {
    match lock {
        0 => "the node's default (normally 2 weeks, no bonus)",
        1 => "3 months (a reward bonus, about x1.03 in 2026)",
        2 => "6 months (a reward bonus, about x1.085 in 2026)",
        3 => "1 year (a reward bonus, about x1.21 in 2026)",
        _ => "invalid",
    }
}

/// What [`compose_password`] found worth telling the user.
#[derive(Debug, Default, PartialEq)]
pub struct PasswordNotes {
    /// Not fatal, but probably not what was meant.
    pub warnings: Vec<String>,
    /// The `lock` in effect (from `--lock` or `lock=`).
    pub lock: Option<u8>,
    /// The fixed difficulty asked for (`d=`).
    pub difficulty: Option<f64>,
}

const KNOWN: [&str; 7] = [
    "d",
    "frequency",
    "freq",
    "skip",
    "lock",
    "parentupdate",
    "auth",
];

/// Builds the stratum password: `auth=<token>` first (when a token is
/// given), then the options in `pass`, then `lock=<lock>` (when given).
/// Errors are what the node would refuse or silently mangle.
pub fn compose_password(
    pass: &str,
    token: Option<&str>,
    lock: Option<u8>,
) -> Result<(String, PasswordNotes), String> {
    let mut notes = PasswordNotes::default();
    if let Some(t) = token
        && (t.contains(',') || t.contains('_'))
    {
        return Err(
            "the auth token contains ',' or '_', which separate password options: it can never match"
                .into(),
        );
    }
    let pass = pass.trim();
    let opts: Vec<&str> = if pass.is_empty() || pass == "x" {
        Vec::new()
    } else {
        pass.split([',', '_']).collect()
    };
    for o in &opts {
        let (k, v) = o.split_once('=').unwrap_or((o, ""));
        match k {
            "d" => {
                let d: f64 = v.parse().map_err(|_| {
                    if v.contains('=') {
                        format!("{o}: separate password options with ',' (not '.')")
                    } else {
                        format!("d={v}: not a number")
                    }
                })?;
                if !(d.is_finite() && d > 0.0) {
                    return Err(format!("d={v}: must be above 0"));
                }
                if d < 0.1 {
                    notes.warnings.push(format!(
                        "d={v}: Quai nodes raise a KawPoW difficulty below 0.1 to 0.1"
                    ));
                }
                if d > 1e6 {
                    return Err(format!(
                        "d={v}: above 1e6 the node reads it as a hash count, not a difficulty"
                    ));
                }
                notes.difficulty = Some(d);
            }
            "lock" => {
                let l = v
                    .parse::<u8>()
                    .ok()
                    .filter(|l| *l <= 3)
                    .ok_or_else(|| format!("lock={v}: must be 0, 1, 2 or 3"))?;
                if lock.is_some_and(|x| x != l) {
                    return Err(format!("lock={v} in the password disagrees with --lock"));
                }
                notes.lock = Some(l);
            }
            "auth" if token.is_some() => {
                return Err("auth= is in the password and a token file is given too".into());
            }
            k if KNOWN.contains(&k) => {}
            _ => notes.warnings.push(format!(
                "password option {o:?} is not one Quai nodes know (d, frequency, skip, lock, parentupdate, auth); separate options with ','"
            )),
        }
    }
    if let Some(l) = lock {
        if l > 3 {
            return Err(format!("--lock {l}: must be 0, 1, 2 or 3"));
        }
        notes.lock = Some(l);
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = token {
        parts.push(format!("auth={t}"));
    }
    parts.extend(opts.iter().map(|o| (*o).to_string()));
    if let Some(l) = lock
        && !opts.iter().any(|o| o.starts_with("lock="))
    {
        parts.push(format!("lock={l}"));
    }
    Ok((parts.join(","), notes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledgers_and_zones() {
        // The rig's Qi address and a Quai one, both cyprus1.
        let qi = parse_payout("0x00b9A46D00f00aE0b4e21b07db6AB3864b4B135f.octo0").unwrap();
        assert_eq!((qi.ledger, qi.zone().as_str()), (Ledger::Qi, "cyprus1"));
        let quai = parse_payout("0x00050723a3F8e87f704bb78014FE7791FAf344F5").unwrap();
        assert_eq!(quai.ledger, Ledger::Quai);
        assert_eq!(zone_name(0x11), "paxos2");
        assert_eq!(zone_name(0x22), "hydra3");
        assert_eq!(zone_name(0x35), "region 3 zone 5");
        // Short, long, not hex, no 0x.
        assert!(parse_payout("0x00b9A46D00f00aE0b4e21b07db6AB3864b4B13").is_err());
        assert!(parse_payout("0x00b9A46D00f00aE0b4e21b07db6AB3864b4B135f00").is_err());
        assert!(parse_payout("0x00b9A46D00f00aE0b4e21b07db6AB3864b4B135g").is_err());
        assert!(parse_payout("00b9A46D00f00aE0b4e21b07db6AB3864b4B135f").is_err());
    }

    #[test]
    fn workers() {
        assert!(check_worker("0xab.rig-1_a").is_ok());
        assert!(check_worker("0xab").is_ok());
        assert!(check_worker("0xab.rig.gpu0").is_err());
        assert!(check_worker("0xab.").is_err());
        assert!(check_worker("0xab.rig 1").is_err());
    }

    #[test]
    fn passwords() {
        let (p, n) = compose_password("d=1", Some("tok"), Some(2)).unwrap();
        assert_eq!(p, "auth=tok,d=1,lock=2");
        assert_eq!((n.lock, n.difficulty), (Some(2), Some(1.0)));
        // lock= in the password alone, '_' separators, and nothing at all.
        let (p, n) = compose_password("d=0.2_lock=3", None, None).unwrap();
        assert_eq!((p.as_str(), n.lock), ("d=0.2,lock=3", Some(3)));
        assert_eq!(compose_password("x", None, None).unwrap().0, "");
        // What the node would refuse or mangle.
        assert!(compose_password("", Some("a_b"), None).is_err());
        assert!(compose_password("lock=4", None, None).is_err());
        assert!(compose_password("lock=1", None, Some(2)).is_err());
        assert!(compose_password("", None, Some(5)).is_err());
        assert!(compose_password("d=abc", None, None).is_err());
        assert!(compose_password("d=2e6", None, None).is_err());
        assert!(compose_password("auth=x", Some("y"), None).is_err());
        // Accepted with a warning.
        let (_, n) = compose_password("d=0.01", None, None).unwrap();
        assert_eq!(n.warnings.len(), 1);
        // The '.' separator old Quai docs showed.
        assert!(
            compose_password("d=1.lock=2", None, None)
                .unwrap_err()
                .contains("','")
        );
        let (_, n) = compose_password("d=1,colour=red", None, None).unwrap();
        assert_eq!(n.warnings.len(), 1);
    }
}
