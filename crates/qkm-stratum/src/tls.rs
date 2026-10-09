//! TLS for `stratum+ssl://` pools (rustls with ring).
//!
//! Quai nodes serve stratum in the clear, so TLS usually means a proxy in
//! front of the node (stunnel, nginx, HAProxy) with a self-signed
//! certificate. The server's certificate is checked against Mozilla's CAs
//! by default, against a CA file, or against a pinned SHA-256 fingerprint
//! (the self-signed case). When a certificate is refused, the error names
//! its fingerprint so it can be pinned.
//!
//! One thread reads and another writes, as on a plain socket: the TLS state
//! sits behind a mutex that is never held while waiting for the pool.

use std::fmt;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rustls::client::WebPkiServerVerifier;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, verify_tls12_signature, verify_tls13_signature};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{
    CertificateError, ClientConfig, ClientConnection, DigitallySignedStruct, RootCertStore,
    SignatureScheme,
};

/// URL schemes for TLS pools.
pub const TLS_SCHEMES: [&str; 2] = ["stratum+ssl", "stratum+tls"];
/// URL schemes for plain pools (a URL may also have none).
pub const PLAIN_SCHEMES: [&str; 1] = ["stratum+tcp"];

/// Whether `url` asks for TLS (`stratum+ssl://` or `stratum+tls://`);
/// `stratum+tcp://` or no scheme is plain TCP, anything else an error.
pub fn wants_tls(url: &str) -> Result<bool, String> {
    let Some((scheme, _)) = url.trim().split_once("://") else {
        return Ok(false);
    };
    let s = scheme.to_ascii_lowercase();
    if TLS_SCHEMES.contains(&s.as_str()) {
        Ok(true)
    } else if PLAIN_SCHEMES.contains(&s.as_str()) {
        Ok(false)
    } else {
        Err(format!(
            "{url}: unknown scheme {scheme}:// (use stratum+tcp:// or stratum+ssl://)"
        ))
    }
}

/// Which certificate a TLS pool may present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Trust {
    /// One for the pool's host name, from a CA in Mozilla's list.
    Public,
    /// One for the pool's host name, from a CA in this PEM file.
    CaFile(PathBuf),
    /// Exactly this certificate (the SHA-256 of its DER), whatever its
    /// name, issuer or dates: for a self-signed certificate.
    Pinned([u8; 32]),
}

impl fmt::Display for Trust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Trust::Public => write!(f, "public CAs"),
            Trust::CaFile(p) => write!(f, "CA file {}", p.display()),
            Trust::Pinned(fp) => write!(f, "pinned certificate {}", fingerprint_text(fp)),
        }
    }
}

/// A TLS pool's settings, checked: [`Tls::new`] has read the CA file.
#[derive(Clone)]
pub struct Tls {
    trust: Trust,
    config: Arc<ClientConfig>,
    /// The fingerprint of the last certificate a server presented.
    seen: Arc<Mutex<Option<[u8; 32]>>>,
}

impl fmt::Debug for Tls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tls").field("trust", &self.trust).finish()
    }
}

impl PartialEq for Tls {
    fn eq(&self, other: &Tls) -> bool {
        self.trust == other.trust
    }
}

impl Eq for Tls {}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// SHA-256 of a certificate's DER.
pub fn fingerprint(der: &[u8]) -> [u8; 32] {
    let d = ring::digest::digest(&ring::digest::SHA256, der);
    let mut out = [0u8; 32];
    out.copy_from_slice(d.as_ref());
    out
}

/// `AB:CD:…`, as `openssl x509 -noout -fingerprint -sha256` prints it.
pub fn fingerprint_text(fp: &[u8; 32]) -> String {
    fp.iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(":")
}

/// Reads a SHA-256 fingerprint: 64 hex digits, with or without colons,
/// optionally after `sha256:` or openssl's `sha256 Fingerprint=`.
pub fn parse_fingerprint(s: &str) -> Result<[u8; 32], String> {
    let t = s.trim();
    let t = t.rsplit_once('=').map_or(t, |(_, v)| v);
    let t = t
        .strip_prefix("sha256:")
        .or_else(|| t.strip_prefix("SHA256:"))
        .unwrap_or(t);
    let hex: String = t.chars().filter(|c| !matches!(c, ':' | ' ')).collect();
    let bytes = crate::proto::unhex(&hex)
        .ok()
        .filter(|b| b.len() == 32)
        .ok_or_else(|| {
            format!("{s:?} is not a SHA-256 fingerprint (64 hex digits, colons allowed)")
        })?;
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

impl Tls {
    /// Settings for `trust` (reads a CA file now, so a bad one fails early).
    pub fn new(trust: Trust) -> Result<Tls, String> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let check: Arc<dyn ServerCertVerifier> = match &trust {
            Trust::Public => {
                let roots = RootCertStore {
                    roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
                };
                webpki(roots, &provider)?
            }
            Trust::CaFile(path) => {
                let mut roots = RootCertStore::empty();
                let certs = CertificateDer::pem_file_iter(path)
                    .map_err(|e| format!("cannot read the CA file {}: {e}", path.display()))?;
                for c in certs {
                    let c =
                        c.map_err(|e| format!("bad PEM in the CA file {}: {e}", path.display()))?;
                    roots.add(c).map_err(|e| {
                        format!("bad certificate in the CA file {}: {e}", path.display())
                    })?;
                }
                if roots.is_empty() {
                    return Err(format!("no certificate in the CA file {}", path.display()));
                }
                webpki(roots, &provider)?
            }
            Trust::Pinned(pin) => Arc::new(Pinned {
                pin: *pin,
                provider: provider.clone(),
            }),
        };
        let seen = Arc::new(Mutex::new(None));
        let verifier = Arc::new(Recorder {
            inner: check,
            seen: seen.clone(),
        });
        let config = ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| format!("TLS setup: {e}"))?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        Ok(Tls {
            trust,
            config: Arc::new(config),
            seen,
        })
    }

    /// What the server's certificate is checked against.
    pub fn trust(&self) -> &Trust {
        &self.trust
    }

    /// Runs the handshake over `sock` for `host` (a name or an IP address),
    /// giving each read and write `timeout`.
    pub(crate) fn connect(
        &self,
        host: &str,
        mut sock: TcpStream,
        timeout: Duration,
    ) -> Result<Session, String> {
        let name = ServerName::try_from(host.to_string())
            .map_err(|e| format!("TLS: {host:?} is not a host name: {e}"))?;
        let mut conn = ClientConnection::new(self.config.clone(), name)
            .map_err(|e| format!("TLS setup: {e}"))?;
        *lock(&self.seen) = None;
        let _ = sock.set_read_timeout(Some(timeout));
        let _ = sock.set_write_timeout(Some(timeout));
        while conn.is_handshaking() {
            if let Err(e) = conn.complete_io(&mut sock) {
                let _ = sock.shutdown(Shutdown::Both);
                return Err(self.handshake_error(&e));
            }
        }
        Ok(Session {
            conn: Arc::new(Mutex::new(conn)),
            sock,
        })
    }

    fn handshake_error(&self, e: &io::Error) -> String {
        let tls = e.get_ref().and_then(|i| i.downcast_ref::<rustls::Error>());
        let mut msg = match tls {
            Some(t) => format!("TLS handshake: {t}"),
            None if matches!(
                e.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ) =>
            {
                "TLS handshake: no answer (is the pool serving TLS?)".to_string()
            }
            None => format!("TLS handshake: {e}"),
        };
        if let Some(rustls::Error::InvalidCertificate(_)) = tls
            && let Some(fp) = *lock(&self.seen)
        {
            msg.push_str(&format!(
                " (checked against {}); the pool's certificate is SHA-256 {}: if that is \
                 your proxy's certificate, pin it with tls_fingerprint",
                self.trust,
                fingerprint_text(&fp)
            ));
        }
        msg
    }
}

fn webpki(
    roots: RootCertStore,
    provider: &Arc<CryptoProvider>,
) -> Result<Arc<dyn ServerCertVerifier>, String> {
    let v = WebPkiServerVerifier::builder_with_provider(Arc::new(roots), provider.clone())
        .build()
        .map_err(|e| format!("TLS setup: {e}"))?;
    Ok(v)
}

/// Notes each certificate's fingerprint, then lets `inner` decide.
#[derive(Debug)]
struct Recorder {
    inner: Arc<dyn ServerCertVerifier>,
    seen: Arc<Mutex<Option<[u8; 32]>>>,
}

impl ServerCertVerifier for Recorder {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        *lock(&self.seen) = Some(fingerprint(end_entity));
        self.inner
            .verify_server_cert(end_entity, intermediates, server_name, ocsp_response, now)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

/// Accepts exactly one certificate. The handshake signature is still
/// checked against it, so the server must hold its key.
#[derive(Debug)]
struct Pinned {
    pin: [u8; 32],
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        if fingerprint(end_entity) == self.pin {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(rustls::Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// An established TLS connection, to be split into a [`Reader`] and a
/// [`Writer`].
pub(crate) struct Session {
    conn: Arc<Mutex<ClientConnection>>,
    sock: TcpStream,
}

impl Session {
    /// The socket under the TLS (for its options).
    pub(crate) fn socket(&self) -> &TcpStream {
        &self.sock
    }

    /// Splits the connection; `read_half` is a clone of the socket.
    pub(crate) fn split(self, read_half: TcpStream) -> (Reader, Writer) {
        (
            Reader {
                conn: self.conn.clone(),
                sock: read_half,
            },
            Writer {
                conn: self.conn,
                sock: self.sock,
            },
        )
    }
}

/// Plaintext from the pool.
pub(crate) struct Reader {
    conn: Arc<Mutex<ClientConnection>>,
    sock: TcpStream,
}

impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            {
                let mut c = lock(&self.conn);
                match c.reader().read(buf) {
                    // 0: the pool closed the TLS session (close_notify).
                    Ok(n) => return Ok(n),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e),
                }
            }
            // Wait for the pool without holding the lock (the writer needs
            // it); the socket's read timeout applies.
            self.sock.peek(&mut [0u8; 1])?;
            let mut c = lock(&self.conn);
            // At end of stream this reads 0, and the reader above then
            // tells a clean close from an abrupt one.
            c.read_tls(&mut self.sock)?;
            if let Err(e) = c.process_new_packets() {
                let _ = c.write_tls(&mut self.sock);
                return Err(io::Error::new(io::ErrorKind::InvalidData, e));
            }
            // Anything the session has to answer (e.g. a key update).
            while c.wants_write() {
                c.write_tls(&mut self.sock)?;
            }
        }
    }
}

/// Plaintext to the pool.
pub(crate) struct Writer {
    conn: Arc<Mutex<ClientConnection>>,
    sock: TcpStream,
}

impl Writer {
    /// Says goodbye (close_notify) and closes the socket, which also ends
    /// the reader.
    pub(crate) fn close(&mut self) {
        {
            let mut c = lock(&self.conn);
            c.send_close_notify();
            while c.wants_write() {
                if c.write_tls(&mut self.sock).is_err() {
                    break;
                }
            }
        }
        let _ = self.sock.shutdown(Shutdown::Both);
    }
}

impl Write for Writer {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut c = lock(&self.conn);
        let n = c.writer().write(buf)?;
        while c.wants_write() {
            c.write_tls(&mut self.sock)?;
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schemes() {
        assert_eq!(wants_tls("10.0.0.13:3333"), Ok(false));
        assert_eq!(wants_tls("stratum+tcp://h:1"), Ok(false));
        assert_eq!(wants_tls("stratum+ssl://h:1"), Ok(true));
        assert_eq!(wants_tls("STRATUM+TLS://h:1"), Ok(true));
        assert!(wants_tls("stratum2+tcp://h:1").is_err());
    }

    #[test]
    fn fingerprints() {
        let fp: [u8; 32] = std::array::from_fn(|i| i as u8 * 7);
        let text = fingerprint_text(&fp);
        assert_eq!(text.len(), 32 * 3 - 1);
        assert_eq!(parse_fingerprint(&text), Ok(fp));
        assert_eq!(
            parse_fingerprint(&text.replace(':', "").to_lowercase()),
            Ok(fp)
        );
        assert_eq!(
            parse_fingerprint(&format!("sha256 Fingerprint={text}")),
            Ok(fp)
        );
        assert_eq!(parse_fingerprint(&format!("sha256:{text}")), Ok(fp));
        assert!(parse_fingerprint(&text[3..]).is_err());
        assert!(parse_fingerprint("zz").is_err());
        assert_eq!(
            Trust::Pinned(fp).to_string(),
            format!("pinned certificate {text}")
        );
    }

    #[test]
    fn a_missing_ca_file_fails_early() {
        let e = Tls::new(Trust::CaFile("/nonexistent/ca.pem".into())).unwrap_err();
        assert!(e.contains("/nonexistent/ca.pem"), "{e}");
        assert!(Tls::new(Trust::Public).is_ok());
    }
}
