//! The full-screen view of `mine`: the rig at a glance (hashrate and its
//! last ten minutes, shares, pool, job, power and efficiency), a row per GPU
//! with everything nvidia-smi would show (temperature, fan, power, clocks,
//! utilization, VRAM, P-state, PCIe, why the clocks are held down) next to
//! its hashrate and shares, and the log.
//!
//! Keys: `q` twice (or Ctrl-C) quits; arrows, PgUp/PgDn and Home/End scroll
//! the log (End follows it again).
//!
//! While the TUI is up, stderr goes to the log file (or /dev/null) so that
//! nothing writes over the screen.

use std::fs::File;
use std::io::{self, Stdout};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::{cursor, execute, terminal};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Sparkline, Table};
use ratatui::Frame;

use crate::stats::{GpuState, GpuView, Stats, View};

/// The saved stderr while it is redirected (-1 when it is not).
static SAVED_STDERR: AtomicI32 = AtomicI32::new(-1);

fn restore_stderr() {
    let fd = SAVED_STDERR.swap(-1, Ordering::SeqCst);
    if fd >= 0 {
        // SAFETY: `fd` is the duplicate of stderr taken in `redirect_stderr`.
        unsafe {
            libc::dup2(fd, 2);
            libc::close(fd);
        }
    }
}

fn redirect_stderr(to: Option<&Path>) {
    let f = match to {
        Some(p) => std::fs::OpenOptions::new().create(true).append(true).open(p),
        None => File::options().write(true).open("/dev/null"),
    };
    let Ok(f) = f else { return };
    // SAFETY: plain descriptor duplication; on failure stderr is left as is.
    unsafe {
        let saved = libc::dup(2);
        if saved < 0 {
            return;
        }
        if libc::dup2(f.as_raw_fd(), 2) < 0 {
            libc::close(saved);
            return;
        }
        SAVED_STDERR.store(saved, Ordering::SeqCst);
    }
}

fn restore_terminal() {
    let _ = terminal::disable_raw_mode();
    let _ = execute!(io::stdout(), terminal::LeaveAlternateScreen, cursor::Show);
    restore_stderr();
}

/// Takes over the terminal and draws until `stop` is set; `q` twice or
/// Ctrl-C sets it. `log_file` receives stderr meanwhile.
pub fn start(
    stats: Arc<Stats>,
    stop: Arc<AtomicBool>,
    log_file: Option<&Path>,
) -> io::Result<JoinHandle<()>> {
    terminal::enable_raw_mode()?;
    if let Err(e) = execute!(io::stdout(), terminal::EnterAlternateScreen, cursor::Hide) {
        let _ = terminal::disable_raw_mode();
        return Err(e);
    }
    let term = match Terminal::new(CrosstermBackend::new(io::stdout())) {
        Ok(t) => t,
        Err(e) => {
            restore_terminal();
            return Err(e);
        }
    };
    // A panic anywhere puts the terminal back before the message prints.
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        hook(info);
    }));
    redirect_stderr(log_file);
    std::thread::Builder::new()
        .name("tui".into())
        .spawn(move || {
            ui_loop(term, &stats, &stop);
            restore_terminal();
        })
}

/// What the keys changed.
#[derive(Default)]
struct UiState {
    /// Log lines scrolled back from the end (0 follows the log).
    scroll: usize,
    /// When `q` was pressed once (a second press within 3 s quits).
    quit_armed: Option<Instant>,
}

/// Whether the terminal is gone (its window or SSH session closed): the
/// tty then reads as end-of-file forever, and crossterm's `read` would spin
/// on it without returning.
fn terminal_gone() -> bool {
    let mut p = libc::pollfd {
        fd: 0,
        events: libc::POLLIN,
        revents: 0,
    };
    // SAFETY: one valid pollfd, no wait.
    let n = unsafe { libc::poll(&mut p, 1, 0) };
    n < 0 || p.revents & (libc::POLLHUP | libc::POLLERR | libc::POLLNVAL) != 0
}

fn ui_loop(mut term: Terminal<CrosstermBackend<Stdout>>, stats: &Stats, stop: &AtomicBool) {
    let mut ui = UiState::default();
    let mut last_draw = Instant::now() - Duration::from_secs(10);
    let mut dirty = true;
    while !stop.load(Ordering::SeqCst) {
        // A lost terminal stops the miner, as Ctrl-C would (SIGHUP does too,
        // but the input reader must not be touched once the tty is gone).
        if terminal_gone() {
            crate::output::event(format!(
                "{} the terminal closed; stopping",
                crate::miner::now_hms()
            ));
            stop.store(true, Ordering::SeqCst);
            return;
        }
        if dirty || last_draw.elapsed() >= Duration::from_secs(1) {
            let view = stats.view();
            let log = crate::output::recent();
            if ui.quit_armed.is_some_and(|t| t.elapsed() > Duration::from_secs(3)) {
                ui.quit_armed = None;
            }
            let _ = term.draw(|f| draw(f, &view, &log, &ui));
            last_draw = Instant::now();
            dirty = false;
        }
        if !event::poll(Duration::from_millis(200)).unwrap_or(false) {
            continue;
        }
        let Ok(ev) = event::read() else { continue };
        dirty = true;
        let Event::Key(k) = ev else { continue };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        match k.code {
            KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                stop.store(true, Ordering::SeqCst);
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                if ui.quit_armed.is_some() {
                    stop.store(true, Ordering::SeqCst);
                } else {
                    ui.quit_armed = Some(Instant::now());
                }
            }
            KeyCode::Esc => ui.quit_armed = None,
            KeyCode::Up => ui.scroll += 1,
            KeyCode::Down => ui.scroll = ui.scroll.saturating_sub(1),
            KeyCode::PageUp => ui.scroll += 10,
            KeyCode::PageDown => ui.scroll = ui.scroll.saturating_sub(10),
            KeyCode::Home => ui.scroll = usize::MAX / 2,
            KeyCode::End => ui.scroll = 0,
            _ => {}
        }
    }
    if terminal_gone() {
        return;
    }
    // One last frame so the screen shows the miner stopping.
    let view = stats.view();
    let log = crate::output::recent();
    let _ = term.draw(|f| draw(f, &view, &log, &ui));
}

// ---- formatting ----

fn mhs(h: f64) -> String {
    format!("{:.2}", h / 1e6)
}

/// `12s`, `4m05s`, `2h13m`, `3d04h`.
fn age(d: Duration) -> String {
    let s = d.as_secs();
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m{:02}s", s / 60, s % 60),
        3600..86400 => format!("{}h{:02}m", s / 3600, s / 60 % 60),
        _ => format!("{}d{:02}h", s / 86400, s / 3600 % 24),
    }
}

fn uptime(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 86400 {
        format!("{}d {:02}:{:02}:{:02}", s / 86400, s / 3600 % 24, s / 60 % 60, s % 60)
    } else {
        format!("{:02}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    }
}

/// `0x00b9A4…135f.octo0`: the address shortened, the worker name kept.
fn short_worker(w: &str) -> String {
    let (addr, name) = w.split_once('.').unwrap_or((w, ""));
    let chars: Vec<char> = addr.chars().collect();
    let addr = if chars.len() > 14 {
        let head: String = chars[..8].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}…{tail}")
    } else {
        addr.to_string()
    };
    if name.is_empty() { addr } else { format!("{addr}.{name}") }
}

/// `NVIDIA GeForce RTX 3060 Ti` → `RTX 3060 Ti`.
fn short_name(n: &str) -> &str {
    n.trim_start_matches("NVIDIA ")
        .trim_start_matches("GeForce ")
}

fn temp_style(t: u32) -> Style {
    match t {
        0..70 => Style::new().fg(Color::Green),
        70..80 => Style::new().fg(Color::Yellow),
        _ => Style::new().fg(Color::Red).add_modifier(Modifier::BOLD),
    }
}

fn state_cell(g: &GpuView, connected: bool) -> Cell<'static> {
    match &g.state {
        GpuState::Mining if g.rate_1m <= 0.0 => Cell::from("starting").fg(Color::Yellow),
        GpuState::Mining => Cell::from("mining").fg(Color::Green),
        GpuState::BuildingDag(e) => Cell::from(format!("DAG {e}")).fg(Color::Yellow),
        GpuState::Error => Cell::from("error").red().bold(),
        GpuState::Failed => Cell::from("failed").red().bold(),
        GpuState::TooHot => Cell::from("too hot").red().bold(),
        GpuState::Idle if connected => Cell::from("idle").fg(Color::Red),
        GpuState::Idle => Cell::from("idle").fg(Color::DarkGray),
    }
}

fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

/// The rig's power draw, when every GPU reports one.
fn total_power(v: &View) -> Option<f64> {
    v.gpus.iter().map(|g| g.reading.power_w).sum()
}

// ---- drawing ----

fn draw(f: &mut Frame, v: &View, log: &[String], ui: &UiState) {
    let area = f.area();
    let table_h = (v.gpus.len() as u16 + 4).min(area.height.saturating_sub(12).max(5));
    let [header, summary, gpus, logs, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(6),
        Constraint::Length(table_h),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .areas(area);
    draw_header(f, header, v);
    draw_summary(f, summary, v);
    draw_gpus(f, gpus, v);
    draw_log(f, logs, log, ui.scroll);
    draw_footer(f, footer, ui);
}

fn draw_header(f: &mut Frame, area: Rect, v: &View) {
    let (dot, pool) = if v.connected {
        ("●", Span::raw(v.pool.clone()).green())
    } else {
        ("○", Span::raw("not connected").red().bold())
    };
    let wide = area.width >= 100;
    let sep = if wide { "  │  " } else { " │ " };
    let mut spans = vec![Span::raw(if wide { " quai-kawpow-miner " } else { " qkm" }).bold()];
    if wide {
        spans.push(Span::raw(env!("CARGO_PKG_VERSION")).fg(Color::Gray));
    }
    // Narrow: the worker name alone (the address is in the log).
    let who = if wide {
        short_worker(&v.worker)
    } else {
        v.worker.split_once('.').map_or_else(|| short_worker(&v.worker), |(_, w)| w.to_string())
    };
    spans.extend([Span::raw(sep), Span::raw(who).bold()]);
    if !v.payout.is_empty() {
        spans.push(Span::raw(format!(" ({})", v.payout)).fg(Color::Yellow));
    }
    spans.extend([
        Span::raw(sep),
        Span::raw(format!("{dot} ")).fg(if v.connected { Color::Green } else { Color::Red }),
        pool,
        Span::raw(sep),
        Span::raw("up "),
        Span::raw(uptime(v.uptime)),
    ]);
    if wide {
        spans.push(Span::raw(format!(
            "{sep}{} GPU{}",
            v.gpus.len(),
            if v.gpus.len() == 1 { "" } else { "s" }
        )));
    }
    let line = Line::from(spans);
    f.render_widget(
        Paragraph::new(line).style(Style::new().bg(Color::Indexed(236)).fg(Color::White)),
        area,
    );
}

fn draw_summary(f: &mut Frame, area: Rect, v: &View) {
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(45), Constraint::Percentage(55)]).areas(area);

    // Hashrate and its last ten minutes.
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(" Hashrate ".bold());
    let inner = block.inner(left);
    f.render_widget(block, left);
    let [num, spark] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);
    let w = spark.width as usize;
    let recent = &v.history[v.history.len().saturating_sub(w)..];
    let lo = recent.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = recent.iter().copied().fold(0.0, f64::max);
    let mut top = vec![
        Span::raw(format!("{} MH/s", mhs(v.rate_1m))).bold().cyan(),
        Span::styled(format!("  10 min {}", mhs(v.rate_10m)), dim()),
    ];
    if !recent.is_empty() && inner.width >= 52 {
        top.push(Span::styled(format!("  range {}–{}", mhs(lo), mhs(hi)), dim()));
    }
    f.render_widget(Paragraph::new(Line::from(top)), num);
    // Scaled to the recent range (a steady rig is not a solid wall), with
    // some room below so a dip shows.
    let pad = (hi - lo).max(hi * 0.02);
    let floor = (lo - pad).max(0.0);
    let pts: Vec<u64> = recent.iter().map(|h| ((h - floor) / 1e3) as u64).collect();
    f.render_widget(
        Sparkline::default()
            .data(&pts)
            .max(((hi + pad * 0.25 - floor) / 1e3).max(1.0) as u64)
            .style(Style::new().fg(Color::Cyan)),
        spark,
    );

    // Shares, job, pool, power.
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(" Shares & pool ".bold());
    let inner = block.inner(right);
    f.render_widget(block, right);
    let bad = |n: u64, c: Color| {
        if n > 0 {
            Style::new().fg(c).bold()
        } else {
            dim()
        }
    };
    let mut shares = vec![
        Span::raw("Shares  "),
        Span::raw(format!("A {}", v.accepted)).green().bold(),
        Span::raw("  "),
        Span::styled(format!("R {}", v.rejected), bad(v.rejected, Color::Red)),
        Span::raw("  "),
        Span::styled(format!("S {}", v.stale), bad(v.stale, Color::Yellow)),
        Span::raw("  "),
        Span::styled(format!("L {}", v.lost), bad(v.lost, Color::Yellow)),
        Span::raw("   last "),
        Span::raw(v.last_accept.map_or_else(|| "-".into(), age)),
    ];
    if v.gpu_errors > 0 {
        shares.push(Span::raw(format!("   GPU ERRORS {}", v.gpu_errors)).red().bold());
    }
    let job = match &v.job {
        Some((_, h)) => format!("h{h}  epoch {}", h / 7500),
        None => "-".into(),
    };
    let mut lines = vec![
        Line::from(shares),
        Line::from(vec![
            Span::raw("Job     "),
            Span::raw(job),
            Span::raw(format!("   diff {}", v.difficulty)),
        ]),
        Line::from(vec![
            Span::raw("Latency "),
            Span::raw(v.avg_latency_ms.map_or_else(|| "-".into(), |l| format!("{l} ms"))),
            Span::raw("   Power "),
            Span::raw(total_power(v).map_or_else(|| "-".into(), |p| format!("{p:.0} W"))),
            Span::raw("   "),
            Span::raw(
                total_power(v)
                    .filter(|p| *p > 0.0)
                    .map_or_else(String::new, |p| format!("{:.1} kH/W", v.rate_1m / 1e3 / p)),
            ),
        ]),
    ];
    if let Some(r) = &v.last_reject {
        lines.push(Line::from(vec![
            Span::raw("Last refusal "),
            Span::raw(r.clone()).yellow(),
        ]));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// A table column: header, width, and priority (lower stays on narrow
/// screens).
struct Col {
    name: &'static str,
    width: u16,
    prio: u8,
}

const COLS: &[Col] = &[
    Col { name: "GPU", width: 3, prio: 0 },
    Col { name: "Name", width: 12, prio: 8 },
    Col { name: "Bus", width: 7, prio: 9 },
    Col { name: "State", width: 8, prio: 1 },
    Col { name: "MH/s", width: 7, prio: 0 },
    Col { name: "10 min", width: 7, prio: 5 },
    Col { name: "A/R", width: 7, prio: 1 },
    Col { name: "Last", width: 6, prio: 4 },
    Col { name: "Temp", width: 4, prio: 0 },
    Col { name: "Fan", width: 4, prio: 2 },
    Col { name: "Power", width: 9, prio: 0 },
    Col { name: "kH/W", width: 5, prio: 3 },
    Col { name: "Core", width: 5, prio: 3 },
    Col { name: "Mem", width: 5, prio: 4 },
    Col { name: "Util", width: 4, prio: 6 },
    Col { name: "VRAM", width: 9, prio: 7 },
    Col { name: "P", width: 3, prio: 7 },
    Col { name: "PCIe", width: 5, prio: 9 },
    Col { name: "Throttle", width: 14, prio: 5 },
];

/// The columns that fit in `width`, in table order.
fn columns(width: u16) -> Vec<usize> {
    let mut order: Vec<usize> = (0..COLS.len()).collect();
    order.sort_by_key(|&i| (COLS[i].prio, i));
    let mut used = 0u16;
    let mut keep = Vec::new();
    for i in order {
        let w = COLS[i].width + 1;
        if used + w <= width {
            used += w;
            keep.push(i);
        }
    }
    keep.sort_unstable();
    keep
}

fn gpu_cells(g: &GpuView, connected: bool) -> Vec<Cell<'static>> {
    let r = &g.reading;
    let na = || Cell::from("-").style(dim());
    COLS.iter()
        .map(|c| match c.name {
            "GPU" => Cell::from(g.device.to_string()).bold(),
            "Name" => Cell::from(short_name(&g.name).to_string()),
            "Bus" => Cell::from(g.bus.split_once(':').map_or(&*g.bus, |b| b.1).to_string())
                .style(dim()),
            "State" => state_cell(g, connected),
            "MH/s" => Cell::from(mhs(g.rate_1m)).bold().cyan(),
            "10 min" => Cell::from(mhs(g.rate_10m)),
            "A/R" => {
                let s = format!("{}/{}", g.accepted, g.refused);
                if g.refused > 0 {
                    Cell::from(s).yellow()
                } else {
                    Cell::from(s)
                }
            }
            "Last" => g
                .last_share
                .map_or_else(na, |d| Cell::from(age(d)).style(dim())),
            "Temp" => r
                .temperature_c
                .map_or_else(na, |t| Cell::from(format!("{t}°")).style(temp_style(t))),
            "Fan" => r.fan_pct.map_or_else(na, |p| {
                let c = Cell::from(format!("{p}%"));
                if p >= 90 { c.yellow() } else { c }
            }),
            "Power" => match (r.power_w, r.power_limit_w) {
                (Some(p), Some(l)) => Cell::from(format!("{p:.0}/{l:.0}W")),
                (Some(p), None) => Cell::from(format!("{p:.0}W")),
                _ => na(),
            },
            "kH/W" => r
                .power_w
                .filter(|p| *p > 1.0)
                .map_or_else(na, |p| Cell::from(format!("{:.0}", g.rate_1m / 1e3 / p))),
            "Core" => r.sm_clock_mhz.map_or_else(na, |m| Cell::from(m.to_string())),
            "Mem" => r.mem_clock_mhz.map_or_else(na, |m| Cell::from(m.to_string())),
            "Util" => r.util_pct.map_or_else(na, |u| Cell::from(format!("{u}%"))),
            "VRAM" => r.vram.map_or_else(na, |(u, t)| {
                let gib = |b: u64| b as f64 / f64::from(1u32 << 30);
                Cell::from(format!("{:.1}/{:.1}G", gib(u), gib(t)))
            }),
            "P" => r.pstate.map_or_else(na, |p| Cell::from(format!("P{p}"))),
            "PCIe" => r
                .pcie
                .map_or_else(na, |(gen_, w)| Cell::from(format!("{gen_}x{w}")).style(dim())),
            "Throttle" => {
                if r.throttle.is_empty() {
                    Cell::from("")
                } else {
                    let serious = r.throttle.iter().any(|t| *t != "power cap");
                    let c = Cell::from(r.throttle.join(", "));
                    if serious { c.red().bold() } else { c.style(dim()) }
                }
            }
            _ => na(),
        })
        .collect()
}

fn total_cells(v: &View) -> Vec<Cell<'static>> {
    let mining = v.gpus.iter().filter(|g| g.state == GpuState::Mining).count();
    let power = total_power(v);
    let acc: u64 = v.gpus.iter().map(|g| g.accepted).sum();
    let refused: u64 = v.gpus.iter().map(|g| g.refused).sum();
    COLS.iter()
        .map(|c| match c.name {
            "GPU" => Cell::from("all").bold(),
            "State" => Cell::from(format!("{mining}/{}", v.gpus.len())),
            "MH/s" => Cell::from(mhs(v.rate_1m)).bold().cyan(),
            "10 min" => Cell::from(mhs(v.rate_10m)),
            "A/R" => Cell::from(format!("{acc}/{refused}")),
            "Power" => Cell::from(power.map_or_else(String::new, |p| format!("{p:.0}W"))).bold(),
            "kH/W" => Cell::from(
                power
                    .filter(|p| *p > 1.0)
                    .map_or_else(String::new, |p| format!("{:.0}", v.rate_1m / 1e3 / p)),
            ),
            _ => Cell::from(""),
        })
        .collect()
}

fn draw_gpus(f: &mut Frame, area: Rect, v: &View) {
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(" GPUs ".bold());
    let keep = columns(area.width.saturating_sub(2));
    let pick = |cells: Vec<Cell<'static>>| -> Row<'static> {
        let mut cells: Vec<Option<Cell>> = cells.into_iter().map(Some).collect();
        Row::new(keep.iter().map(|&i| cells[i].take().unwrap_or_default()).collect::<Vec<_>>())
    };
    let header = Row::new(keep.iter().map(|&i| Cell::from(COLS[i].name)).collect::<Vec<_>>())
        .style(Style::new().fg(Color::Gray).add_modifier(Modifier::BOLD | Modifier::UNDERLINED));
    let mut rows: Vec<Row> = v.gpus.iter().map(|g| pick(gpu_cells(g, v.connected))).collect();
    if v.gpus.len() > 1 {
        rows.push(pick(total_cells(v)).style(Style::new().add_modifier(Modifier::BOLD)).top_margin(0));
    }
    let widths: Vec<Constraint> = keep
        .iter()
        .map(|&i| {
            if COLS[i].name == "Throttle" {
                Constraint::Min(COLS[i].width)
            } else {
                Constraint::Length(COLS[i].width)
            }
        })
        .collect();
    f.render_widget(
        Table::new(rows, widths).header(header).block(block).column_spacing(1),
        area,
    );
}

fn log_style(l: &str) -> Style {
    if l.contains("not accepted") || l.contains("BUG") || l.contains("FAILED") || l.contains("error")
    {
        Style::new().fg(Color::Red)
    } else if l.contains("disconnected") || l.contains("refused") || l.contains("retrying") {
        Style::new().fg(Color::Yellow)
    } else if l.contains(" accepted") {
        Style::new().fg(Color::Green)
    } else if l.contains("epoch") || l.contains("tuning") {
        Style::new().fg(Color::Cyan)
    } else {
        Style::new()
    }
}

fn draw_log(f: &mut Frame, area: Rect, log: &[String], scroll: usize) {
    let h = area.height.saturating_sub(2) as usize;
    let max_scroll = log.len().saturating_sub(h);
    let scroll = scroll.min(max_scroll);
    let end = log.len() - scroll;
    let start = end.saturating_sub(h);
    let title = if scroll > 0 {
        format!(" Log (scrolled back {scroll}; End to follow) ")
    } else {
        " Log ".to_string()
    };
    let lines: Vec<Line> = log[start..end]
        .iter()
        .map(|l| Line::styled(l.clone(), log_style(l)))
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(title.bold()),
        ),
        area,
    );
}

fn draw_footer(f: &mut Frame, area: Rect, ui: &UiState) {
    let line = if ui.quit_armed.is_some() {
        Line::from(" Press q again to stop the miner (Esc cancels) ".bold().black().on_yellow())
    } else {
        Line::from(vec![
            Span::raw(" q").bold(),
            Span::raw(" quit   "),
            Span::raw("↑↓ PgUp PgDn Home").bold(),
            Span::raw(" scroll log   "),
            Span::raw("End").bold(),
            Span::raw(" follow"),
        ])
        .style(dim())
    };
    f.render_widget(Paragraph::new(line), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::DeviceReading;
    use ratatui::backend::TestBackend;

    fn view(n: usize) -> View {
        View {
            uptime: Duration::from_secs(3725),
            connected: true,
            pool: "10.0.0.13:3333".into(),
            worker: "0x00b9A46D00f00aE0b4e21b07db6AB3864b4B135f.octo0".into(),
            payout: "Qi · cyprus1 · lock 2".into(),
            job: Some(("j1".into(), 4_572_712)),
            difficulty: 1.0,
            rate_1m: 145.6e6,
            rate_10m: 145.1e6,
            accepted: 34,
            last_accept: Some(Duration::from_secs(12)),
            avg_latency_ms: Some(14),
            history: (0..120).map(|i| 140e6 + f64::from(i) * 5e4).collect(),
            gpus: (0..n)
                .map(|d| GpuView {
                    device: d,
                    name: "NVIDIA GeForce RTX 3060 Ti".into(),
                    bus: format!("00000000:{:02x}:00.0", d + 1),
                    state: GpuState::Mining,
                    rate_1m: 20.8e6,
                    rate_10m: 20.7e6,
                    accepted: 5,
                    refused: 0,
                    last_share: Some(Duration::from_secs(40)),
                    reading: DeviceReading {
                        temperature_c: Some(60 + d as u32 * 3),
                        power_w: Some(149.8),
                        power_limit_w: Some(150.0),
                        fan_pct: Some(30),
                        mem_clock_mhz: Some(9251),
                        sm_clock_mhz: Some(840),
                        util_pct: Some(100),
                        vram: Some((6_300 << 20, 8_192 << 20)),
                        pstate: Some(2),
                        throttle: vec!["power cap"],
                        pcie: Some((1, 1)),
                    },
                })
                .collect(),
            ..View::default()
        }
    }

    fn render(v: &View, w: u16, h: u16) -> String {
        let mut t = Terminal::new(TestBackend::new(w, h)).unwrap();
        let log = vec![
            "19:16:43 share 34 from gpu3 accepted (14 ms)  A34".to_string(),
            "19:17:02 gpu0: epoch 609: DAG 5.76 GiB built in 10.0 s".to_string(),
        ];
        t.draw(|f| draw(f, v, &log, &UiState::default())).unwrap();
        let buf = t.backend().buffer().clone();
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn wide_screen_shows_everything() {
        let s = render(&view(7), 160, 30);
        if std::env::var_os("QKM_TUI_SHOW").is_some() {
            println!("{s}\n\n{}", render(&view(7), 80, 24));
        }
        for want in [
            "145.60 MH/s", "A 34", "h4572712", "epoch 609", "0x00b9A4…135f.octo0",
            "Qi · cyprus1 · lock 2",
            "RTX 3060 Ti", "150/150W", "9251", "P2", "power cap", "all", "1049 W",
            "share 34 from gpu3", "Log",
        ] {
            assert!(s.contains(want), "missing {want:?} in\n{s}");
        }
    }

    #[test]
    fn narrow_screen_keeps_the_essentials() {
        let s = render(&view(7), 80, 24);
        for want in ["MH/s", "Temp", "Power", "20.80", "150/150W"] {
            assert!(s.contains(want), "missing {want:?} in\n{s}");
        }
        assert!(!s.contains("PCIe"), "a low-priority column on 80 columns:\n{s}");
    }

    #[test]
    fn tiny_screen_does_not_panic() {
        render(&view(7), 30, 8);
        render(&view(1), 1, 1);
    }

    #[test]
    fn ages() {
        assert_eq!(age(Duration::from_secs(12)), "12s");
        assert_eq!(age(Duration::from_secs(245)), "4m05s");
        assert_eq!(age(Duration::from_secs(7980)), "2h13m");
        assert_eq!(short_worker("0xabc.rig"), "0xabc.rig");
        // Not ASCII (once a panic).
        assert_eq!(short_worker("ééééééééééééééééé.r"), "éééééééé…éééé.r");
    }
}
