//! The KawPoW stratum dialect of Quai nodes (rs-quai's `rsq-stratum`, a
//! port of go-quai's): message parsing, request encoding, targets.
//!
//! Server lines are either responses `{"id":N,"result":..,"error":..}` or
//! notifications `{"id":null,"method":..,"params":..}`:
//! - `mining.subscribe` answers `[null, "<extranonce hex>"]` (one byte for
//!   KawPoW), the top byte(s) of every nonce this session submits;
//! - `mining.set_target [target_hex]`, a 256-bit big-endian share target;
//! - `mining.notify [job, header_hash, seed_hash, target, clean, height,
//!   bits]`: the header hash goes to KawPoW byte for byte, `height` is the
//!   Ravencoin donor height (the epoch);
//! - `mining.submit [worker, job, nonce, header_hash, mix]`: `0x`-prefixed
//!   hex; the mix is `kawpow_light`'s mix digest byte for byte (the server
//!   compares it reversed against its own reversed mix).

use serde_json::{Value, json};

/// One KawPoW job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    /// Job id, echoed in `mining.submit`.
    pub id: String,
    /// The header hash KawPoW hashes (bytes as sent).
    pub header: [u8; 32],
    /// The DAG seed hash (32 bytes; checked against the height's epoch).
    pub seed: [u8; 32],
    /// The share target sent with the job (big-endian).
    pub target: Option<[u8; 32]>,
    /// Drop previous jobs.
    pub clean: bool,
    /// Ravencoin donor height.
    pub height: u64,
    /// Compact bits (informational).
    pub bits: String,
}

/// A line from the server.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    /// A response to request `id`.
    Response {
        /// The request id.
        id: u64,
        /// `result`.
        result: Value,
        /// `error` (`null` when none).
        error: Value,
    },
    /// `mining.notify`.
    Notify(Job),
    /// `mining.set_target`.
    SetTarget([u8; 32]),
    /// `mining.set_difficulty` (KawPoW stratum difficulty).
    SetDifficulty(f64),
    /// `mining.set_extranonce [extranonce, ...]`.
    SetExtranonce(Vec<u8>),
    /// `client.reconnect` or any request to reconnect.
    Reconnect,
    /// A notification this client does not use.
    Other(String),
}

/// Parse errors are plain text; a bad line is logged and skipped.
pub type ParseError = String;

/// Longest text from the pool that goes into the log.
pub const MAX_LOGGED: usize = 200;

/// Pool text made safe to log: control characters (terminal escapes) become
/// `?` and it is cut to `max` characters.
pub fn sanitize(s: &str, max: usize) -> String {
    let mut out: String = s
        .chars()
        .take(max)
        .map(|c| if c.is_control() { '?' } else { c })
        .collect();
    if s.chars().nth(max).is_some() {
        out.push('…');
    }
    out
}

/// Hex to bytes, by bytes (pool data: anything but ASCII hex digits is an
/// error, never a panic).
pub fn unhex(s: &str) -> Result<Vec<u8>, ParseError> {
    let t = s.trim();
    let t = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .unwrap_or(t);
    let b = t.as_bytes();
    if !b.len().is_multiple_of(2) {
        return Err(format!("odd-length hex: {}", sanitize(t, 80)));
    }
    let nibble = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    b.chunks_exact(2)
        .map(|p| match (nibble(p[0]), nibble(p[1])) {
            (Some(h), Some(l)) => Ok(h << 4 | l),
            _ => Err(format!("bad hex: {}", sanitize(t, 80))),
        })
        .collect()
}

/// `0x`-prefixed lower-case hex.
pub fn hex0x(b: &[u8]) -> String {
    let mut s = String::with_capacity(2 + 2 * b.len());
    s.push_str("0x");
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn hex32(v: &Value, what: &str) -> Result<[u8; 32], ParseError> {
    let s = v.as_str().ok_or(format!("{what} is not a string"))?;
    let b = unhex(s)?;
    if b.len() > 32 {
        return Err(format!("{what} longer than 32 bytes"));
    }
    // Left-pad short values (a target without leading zeros).
    let mut out = [0u8; 32];
    out[32 - b.len()..].copy_from_slice(&b);
    Ok(out)
}

fn as_u64(v: &Value) -> Option<u64> {
    match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => {
            let t = s.trim();
            t.strip_prefix("0x")
                .map_or_else(|| t.parse().ok(), |h| u64::from_str_radix(h, 16).ok())
        }
        _ => None,
    }
}

/// Parses one line from the server.
pub fn parse_line(line: &str) -> Result<Message, ParseError> {
    let v: Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    if let Some(method) = v.get("method").and_then(Value::as_str) {
        let params = v.get("params").cloned().unwrap_or(Value::Null);
        let p = params.as_array().cloned().unwrap_or_default();
        return match method {
            "mining.notify" => {
                if p.len() < 6 {
                    return Err(format!("mining.notify with {} params", p.len()));
                }
                let id = match &p[0] {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                let target = p.get(3).and_then(|t| hex32(t, "target").ok());
                let height = as_u64(&p[5]).ok_or("bad height")?;
                if !qkm_core::plausible_height(height) {
                    return Err(format!(
                        "height {height} is past epoch {} (a broken or hostile pool)",
                        qkm_core::MAX_EPOCH
                    ));
                }
                Ok(Message::Notify(Job {
                    id,
                    header: hex32(&p[1], "header hash")?,
                    seed: hex32(&p[2], "seed hash")?,
                    target,
                    clean: p.get(4).and_then(Value::as_bool).unwrap_or(false),
                    height,
                    bits: p.get(6).and_then(Value::as_str).unwrap_or("").to_string(),
                }))
            }
            "mining.set_target" => Ok(Message::SetTarget(hex32(
                p.first().ok_or("mining.set_target without a target")?,
                "target",
            )?)),
            "mining.set_difficulty" => p
                .first()
                .and_then(Value::as_f64)
                .filter(|d| d.is_finite() && *d > 0.0)
                .map(Message::SetDifficulty)
                .ok_or_else(|| "bad mining.set_difficulty".to_string()),
            "mining.set_extranonce" => {
                let x = p
                    .first()
                    .and_then(Value::as_str)
                    .ok_or("bad mining.set_extranonce")?;
                Ok(Message::SetExtranonce(unhex(x)?))
            }
            "client.reconnect" => Ok(Message::Reconnect),
            other => Ok(Message::Other(other.to_string())),
        };
    }
    let id = v
        .get("id")
        .and_then(as_u64)
        .ok_or("a response without a numeric id")?;
    Ok(Message::Response {
        id,
        result: v.get("result").cloned().unwrap_or(Value::Null),
        error: v.get("error").cloned().unwrap_or(Value::Null),
    })
}

/// The extranonce in a `mining.subscribe` result (`[null, "c2"]`, or the
/// generic `[[...], "xn1", size]`).
pub fn subscribe_extranonce(result: &Value) -> Vec<u8> {
    result
        .as_array()
        .and_then(|a| a.get(1))
        .and_then(Value::as_str)
        .and_then(|s| unhex(s).ok())
        .unwrap_or_default()
}

/// A request line.
pub fn request(id: u64, method: &str, params: Value) -> String {
    json!({"id": id, "method": method, "params": params}).to_string()
}

/// `mining.submit` for a KawPoW solution.
pub fn submit_params(
    worker: &str,
    job: &str,
    nonce: u64,
    header: &[u8; 32],
    mix: &[u8; 32],
) -> Value {
    json!([
        worker,
        job,
        format!("0x{nonce:016x}"),
        hex0x(header),
        hex0x(mix)
    ])
}

/// The error text of a response (`null` → `None`), safe to log.
pub fn error_text(error: &Value) -> Option<String> {
    let t = match error {
        Value::Null => return None,
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .map(|x| x.as_str().map_or_else(|| x.to_string(), str::to_string))
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    };
    Some(sanitize(&t, MAX_LOGGED))
}

/// KawPoW stratum difficulty 1 (`0x00000000ff000000…`, 2^40/255 hashes).
pub const KAWPOW_DIFF1_TOP: f64 = 255.0;

/// The 256-bit big-endian target of a KawPoW stratum difficulty
/// (`diff1 / difficulty`, `diff1 = 0xff << 216`), rounded down.
pub fn target_for_difficulty(difficulty: f64) -> [u8; 32] {
    if !(difficulty.is_finite() && difficulty > 0.0) {
        return [0xff; 32];
    }
    // t = 255 / d: the target in units of 2^216.
    let t = KAWPOW_DIFF1_TOP / difficulty;
    if t >= 2f64.powi(40) {
        return [0xff; 32];
    }
    // t = m * 2^e with m a 53-bit integer, so target = m << (e + 216).
    let bits = t.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let (m, e) = if exp == 0 {
        (bits & ((1 << 52) - 1), -1074)
    } else {
        ((bits & ((1 << 52) - 1)) | (1 << 52), exp - 1075)
    };
    let shift = e + 216;
    let mut limbs = [0u64; 4]; // little-endian 64-bit limbs
    if shift >= 0 {
        let (q, r) = ((shift / 64) as usize, (shift % 64) as u32);
        if q < 4 {
            limbs[q] |= m << r;
            if r > 0 && q + 1 < 4 {
                limbs[q + 1] |= m >> (64 - r);
            }
        }
    } else if shift > -64 {
        limbs[0] = m >> (-shift);
    }
    let mut out = [0u8; 32];
    for (i, l) in limbs.iter().enumerate() {
        out[24 - 8 * i..32 - 8 * i].copy_from_slice(&l.to_be_bytes());
    }
    out
}

/// The KawPoW stratum difficulty of a target (`diff1 / target`).
pub fn difficulty_of_target(target: &[u8; 32]) -> f64 {
    let mut v = 0f64;
    for b in target {
        v = v * 256.0 + f64::from(*b);
    }
    if v == 0.0 {
        return f64::INFINITY;
    }
    KAWPOW_DIFF1_TOP * 2f64.powi(216) / v
}

/// Replaces the value of `auth=` (any case) in a password or URL with
/// `<redacted>`.
pub fn redact(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    // ASCII lower-casing keeps byte offsets.
    while let Some(i) = rest.to_ascii_lowercase().find("auth=") {
        out.push_str(&rest[..i + 5]);
        out.push_str("<redacted>");
        let tail = &rest[i + 5..];
        let end = tail
            .find([',', '@', '_', ' ', '"', '&'])
            .unwrap_or(tail.len());
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rs_quai_lines() {
        let n = r#"{"id":null,"method":"mining.notify","params":["18db25a5b64bc1030001","aa00000000000000000000000000000000000000000000000000000000000001","bb00000000000000000000000000000000000000000000000000000000000002","00000004faffffffffffec1400000000004faffffffffffec1400000000004fa",true,4565748,"1b068045"]}"#;
        let Message::Notify(j) = parse_line(n).unwrap() else {
            panic!()
        };
        assert_eq!(j.id, "18db25a5b64bc1030001");
        assert_eq!(j.header[0], 0xaa);
        assert_eq!(j.header[31], 0x01);
        assert_eq!(j.height, 4_565_748);
        assert!(j.clean);
        assert_eq!(j.target.unwrap()[3], 0x04);
        let r = parse_line(r#"{"id":3,"result":false,"error":"nojob"}"#).unwrap();
        assert_eq!(
            r,
            Message::Response {
                id: 3,
                result: Value::Bool(false),
                error: json!("nojob")
            }
        );
        let t = parse_line(r#"{"id":null,"method":"mining.set_target","params":["0000000014666666666666666666666666666666666666666666666666666666"]}"#).unwrap();
        let Message::SetTarget(t) = t else { panic!() };
        assert_eq!(t[4], 0x14);
        assert_eq!(subscribe_extranonce(&json!([null, "c2"])), vec![0xc2]);
    }

    #[test]
    fn difficulty_and_target_agree_with_rs_quai() {
        // rs-quai sent 0000000014666666… for vardiff 12.5 (diff1 / 12.5).
        let t = target_for_difficulty(12.5);
        assert_eq!(&t[..6], &[0, 0, 0, 0, 0x14, 0x66]);
        assert!((difficulty_of_target(&t) - 12.5).abs() < 1e-9);
        let t1 = target_for_difficulty(1.0);
        assert_eq!(&t1[..6], &[0, 0, 0, 0, 0xff, 0]);
        assert!((difficulty_of_target(&target_for_difficulty(0.2)) - 0.2).abs() < 1e-9);
    }

    #[test]
    fn redacts_tokens() {
        assert_eq!(redact("auth=abc123,d=0.2"), "auth=<redacted>,d=0.2");
        assert_eq!(
            redact("stratum+tcp://0x00.w:auth=deadbeef@10.0.0.13:3333"),
            "stratum+tcp://0x00.w:auth=<redacted>@10.0.0.13:3333"
        );
        assert_eq!(redact("x"), "x");
        assert_eq!(redact("AUTH=abc,d=1"), "AUTH=<redacted>,d=1");
    }

    #[test]
    fn hostile_lines_are_errors_not_panics() {
        // Multi-byte characters where hex belongs (once a panic).
        let bad = r#"{"id":null,"method":"mining.notify","params":["j","aé00000000000000000000000000000000000000000000000000000000000001","bb","00",true,1,"1b"]}"#;
        assert!(parse_line(bad).is_err());
        assert!(
            parse_line(r#"{"id":null,"method":"mining.set_extranonce","params":["é"]}"#).is_err()
        );
        assert_eq!(subscribe_extranonce(&json!([null, "éé"])), Vec::<u8>::new());
        assert!(unhex("zz").is_err());
        assert_eq!(unhex("0XAbC0").unwrap(), vec![0xab, 0xc0]);
        // A height far past any real epoch.
        let far = format!(
            r#"{{"id":null,"method":"mining.notify","params":["j","{0}","{0}","00",true,{1},"1b"]}}"#,
            "00".repeat(32),
            u64::MAX
        );
        assert!(parse_line(&far).unwrap_err().contains("hostile"));
        // Garbage of every sort.
        for l in [
            "",
            "{",
            "[]",
            "null",
            r#"{"id":"x"}"#,
            r#"{"method":"mining.notify","params":[1,2,3,4,5,6]}"#,
        ] {
            let _ = parse_line(l);
        }
    }

    #[test]
    fn pool_text_is_sanitized() {
        assert_eq!(sanitize("ok\u{1b}[2Jboom", 100), "ok?[2Jboom");
        assert_eq!(sanitize(&"a".repeat(300), 5), "aaaaa…");
        assert_eq!(
            error_text(&json!("\u{1b}]0;pwned\u{7}duplicate share")).unwrap(),
            "?]0;pwned?duplicate share"
        );
    }

    #[test]
    fn target_round_trip_across_the_range() {
        for d in [1e-6, 0.001, 0.1, 0.2, 1.0, 12.5, 1e3, 1e6, 1e9, 1e12] {
            let back = difficulty_of_target(&target_for_difficulty(d));
            assert!((back - d).abs() / d < 1e-9, "{d} -> {back}");
        }
        assert_eq!(target_for_difficulty(0.0), [0xff; 32]);
        assert_eq!(target_for_difficulty(f64::NAN), [0xff; 32]);
    }

    #[test]
    fn submit_wire_format() {
        let p = submit_params("0xabc.w", "j1", 0xc2, &[1; 32], &[2; 32]);
        assert_eq!(p[2], "0x00000000000000c2");
        assert_eq!(p[3].as_str().unwrap().len(), 66);
        assert!(p[4].as_str().unwrap().starts_with("0x0202"));
    }
}
