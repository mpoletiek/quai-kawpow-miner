//! Where the miner's messages go: stdout in plain mode, the log pane in the
//! TUI, and a log file when one is given. Before [`init`], plain stdout.
//!
//! The log file rotates at [`ROTATE_AT`]: `miner.log` becomes `miner.log.1`
//! (and `.1` becomes `.2`, up to [`KEEP_FILES`]).

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Log lines the TUI keeps for its log pane.
const KEEP: usize = 1000;
/// Log file size that triggers a rotation.
pub const ROTATE_AT: u64 = 50 << 20;
/// Rotated log files kept.
pub const KEEP_FILES: u32 = 3;

struct LogFile {
    path: PathBuf,
    file: File,
    size: u64,
}

impl LogFile {
    fn open(path: &Path) -> std::io::Result<LogFile> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let size = file.metadata().map_or(0, |m| m.len());
        Ok(LogFile {
            path: path.to_path_buf(),
            file,
            size,
        })
    }

    fn write(&mut self, text: &str) {
        if self.size >= ROTATE_AT {
            self.rotate();
        }
        if writeln!(self.file, "{text}").is_ok() {
            self.size += text.len() as u64 + 1;
        }
    }

    fn rotate(&mut self) {
        let numbered = |n: u32| {
            let mut p = self.path.clone().into_os_string();
            p.push(format!(".{n}"));
            PathBuf::from(p)
        };
        for n in (1..KEEP_FILES).rev() {
            let _ = std::fs::rename(numbered(n), numbered(n + 1));
        }
        let _ = std::fs::rename(&self.path, numbered(1));
        if let Ok(f) = LogFile::open(&self.path) {
            *self = f;
        } else {
            // Keep writing where we were rather than lose the log.
            self.size = 0;
        }
    }
}

struct Output {
    tui: AtomicBool,
    lines: Mutex<VecDeque<String>>,
    file: Option<Mutex<LogFile>>,
}

static OUT: OnceLock<Output> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets the destination once: stdout, plus `file` (appended to) when given.
/// [`set_tui`] moves the messages to the TUI's log pane.
pub fn init(file: Option<&Path>) -> std::io::Result<()> {
    let file = match file {
        Some(p) => Some(Mutex::new(LogFile::open(p)?)),
        None => None,
    };
    let _ = OUT.set(Output {
        tui: AtomicBool::new(false),
        lines: Mutex::new(VecDeque::new()),
        file,
    });
    Ok(())
}

/// Messages go to the TUI's log pane (`true`) or stdout.
pub fn set_tui(on: bool) {
    if let Some(o) = OUT.get() {
        o.tui.store(on, Ordering::SeqCst);
    }
}

/// Whether the TUI shows the messages (else they go to stdout).
pub fn is_tui() -> bool {
    OUT.get().is_some_and(|o| o.tui.load(Ordering::SeqCst))
}

fn to_file(o: &Output, text: &str) {
    if let Some(f) = &o.file {
        lock(f).write(text);
    }
}

/// An event line (already timestamped).
pub fn event(line: String) {
    let Some(o) = OUT.get() else {
        println!("{line}");
        return;
    };
    to_file(o, &line);
    if o.tui.load(Ordering::SeqCst) {
        let mut l = lock(&o.lines);
        l.push_back(line);
        while l.len() > KEEP {
            l.pop_front();
        }
    } else {
        println!("{line}");
    }
}

/// The periodic status (one or more lines): stdout in plain mode (the TUI
/// shows it live), and the log file either way.
pub fn status(text: &str) {
    let Some(o) = OUT.get() else {
        println!("{text}");
        return;
    };
    to_file(o, text);
    if !o.tui.load(Ordering::SeqCst) {
        println!("{text}");
    }
}

/// The event lines kept for the log pane, oldest first.
pub fn recent() -> Vec<String> {
    OUT.get()
        .map(|o| lock(&o.lines).iter().cloned().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotates() {
        let dir = std::env::temp_dir().join(format!("qkm-log-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("miner.log");
        let mut f = LogFile::open(&path).unwrap();
        for round in 0..5 {
            f.size = ROTATE_AT; // as if full
            f.write(&format!("line {round}"));
        }
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
        assert_eq!(read("miner.log"), "line 4\n");
        assert_eq!(read("miner.log.1"), "line 3\n");
        assert_eq!(read("miner.log.3"), "line 1\n");
        assert!(!dir.join("miner.log.4").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
