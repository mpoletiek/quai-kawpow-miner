//! Where the miner's messages go: stdout in plain mode, the log pane in the
//! TUI, and a log file when one is given. Before [`init`], plain stdout.

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

/// Log lines the TUI keeps for its log pane.
const KEEP: usize = 1000;

struct Output {
    tui: bool,
    lines: Mutex<VecDeque<String>>,
    file: Option<Mutex<File>>,
}

static OUT: OnceLock<Output> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sets the destination once: the TUI's log pane (`tui`) or stdout, plus
/// `file` (appended to) when given.
pub fn init(tui: bool, file: Option<&Path>) -> std::io::Result<()> {
    let file = match file {
        Some(p) => Some(Mutex::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)?,
        )),
        None => None,
    };
    let _ = OUT.set(Output {
        tui,
        lines: Mutex::new(VecDeque::new()),
        file,
    });
    Ok(())
}

/// Whether the TUI shows the messages (else they go to stdout).
pub fn is_tui() -> bool {
    OUT.get().is_some_and(|o| o.tui)
}

fn to_file(o: &Output, text: &str) {
    if let Some(f) = &o.file {
        let _ = writeln!(lock(f), "{text}");
    }
}

/// An event line (already timestamped).
pub fn event(line: String) {
    let Some(o) = OUT.get() else {
        println!("{line}");
        return;
    };
    to_file(o, &line);
    if o.tui {
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
    if !o.tui {
        println!("{text}");
    }
}

/// The event lines kept for the log pane, oldest first.
pub fn recent() -> Vec<String> {
    OUT.get()
        .map(|o| lock(&o.lines).iter().cloned().collect())
        .unwrap_or_default()
}
