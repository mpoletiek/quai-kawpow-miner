//! GPU tuning through NVML: power limit, locked core and memory clocks,
//! clock offsets and fan speed. Setting them needs root; the settings last
//! until reset, a reboot or a driver reload.
//!
//! Locked core clocks also tame the power spikes of a card leaving its power
//! cap (a DAG build ending, work arriving): a rig whose power supplies trip
//! on those spikes can run with them.

use std::collections::BTreeMap;

use nvml_wrapper::Nvml;
use nvml_wrapper::enums::device::GpuLockedClocksSetting;
use nvml_wrapper::error::NvmlError;

/// Settings for one GPU; `None` leaves a setting as it is.
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tuning {
    /// Power limit, watts.
    pub power_limit: Option<u32>,
    /// Core clock locked to `[min, max]` MHz.
    pub lock_core_clocks: Option<[u32; 2]>,
    /// Memory clock locked to `[min, max]` MHz.
    pub lock_mem_clocks: Option<[u32; 2]>,
    /// Core clock offset, MHz.
    pub core_offset: Option<i32>,
    /// Memory clock offset, MHz (as NVML takes it).
    pub mem_offset: Option<i32>,
    /// Fan speed, percent, on every fan of the card (turns automatic fan
    /// control off until reset).
    pub fan: Option<u32>,
}

impl Tuning {
    /// `self` with every setting `over` gives replaced.
    pub fn with(&self, over: &Tuning) -> Tuning {
        Tuning {
            power_limit: over.power_limit.or(self.power_limit),
            lock_core_clocks: over.lock_core_clocks.or(self.lock_core_clocks),
            lock_mem_clocks: over.lock_mem_clocks.or(self.lock_mem_clocks),
            core_offset: over.core_offset.or(self.core_offset),
            mem_offset: over.mem_offset.or(self.mem_offset),
            fan: over.fan.or(self.fan),
        }
    }

    /// Nothing to set.
    pub fn is_empty(&self) -> bool {
        *self == Tuning::default()
    }

    /// Refuses values no card takes.
    pub fn check(&self) -> Result<(), String> {
        for (what, c) in [
            ("core", self.lock_core_clocks),
            ("memory", self.lock_mem_clocks),
        ] {
            if let Some([lo, hi]) = c
                && (lo > hi || hi == 0)
            {
                return Err(format!("locked {what} clocks {lo},{hi}: want MIN,MAX MHz"));
            }
        }
        if self.fan.is_some_and(|f| f > 100) {
            return Err("fan speed above 100%".into());
        }
        Ok(())
    }
}

/// Settings for all GPUs, with per-device overrides (the `[tune]` section
/// of the config, `[tune.gpu.N]` for device N).
#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneConfig {
    /// For every GPU.
    #[serde(flatten)]
    pub all: Tuning,
    /// Overrides by CUDA device index.
    #[serde(default)]
    pub gpu: BTreeMap<String, Tuning>,
}

impl TuneConfig {
    /// The settings for CUDA device `device`.
    pub fn for_device(&self, device: usize) -> Tuning {
        match self.gpu.get(&device.to_string()) {
            Some(o) => self.all.with(o),
            None => self.all.clone(),
        }
    }

    /// `self` with the settings of `over` (command-line flags) on top, for
    /// every GPU.
    pub fn with(&self, over: &Tuning) -> TuneConfig {
        TuneConfig {
            all: self.all.with(over),
            gpu: self
                .gpu
                .iter()
                .map(|(k, t)| (k.clone(), t.with(over)))
                .collect(),
        }
    }

    /// Nothing to set on any GPU.
    pub fn is_empty(&self) -> bool {
        self.all.is_empty() && self.gpu.values().all(Tuning::is_empty)
    }

    /// Checks every setting, and that overrides name devices.
    pub fn check(&self) -> Result<(), String> {
        self.all.check()?;
        for (k, t) in &self.gpu {
            k.parse::<usize>()
                .map_err(|_| format!("[tune.gpu.{k}]: not a device index"))?;
            t.check().map_err(|e| format!("[tune.gpu.{k}]: {e}"))?;
        }
        Ok(())
    }
}

fn why(e: &NvmlError) -> String {
    match e {
        NvmlError::NoPermission => "no permission (needs root)".into(),
        NvmlError::NotSupported => "not supported by this card".into(),
        NvmlError::InvalidArg => "out of the card's range".into(),
        e => e.to_string(),
    }
}

/// The NVML device for the GPU at `pci_bus_id` (`00000000:01:00.0`).
fn device<'a>(nvml: &'a Nvml, pci_bus_id: &str) -> Result<nvml_wrapper::Device<'a>, String> {
    nvml.device_by_pci_bus_id(pci_bus_id)
        .map_err(|e| format!("NVML has no GPU at {pci_bus_id}: {e}"))
}

/// What [`apply`] did.
pub struct Applied {
    /// What was set and what failed, one line.
    pub text: String,
    /// Settings that failed.
    pub failed: usize,
    /// Of those, refused for want of permission (not root).
    pub denied: usize,
}

/// Applies `t` to the GPU at `pci_bus_id`.
pub fn apply(nvml: &Nvml, pci_bus_id: &str, t: &Tuning) -> Applied {
    let mut d = match device(nvml, pci_bus_id) {
        Ok(d) => d,
        Err(e) => {
            return Applied {
                text: e,
                failed: 1,
                denied: 0,
            };
        }
    };
    let mut parts = Vec::new();
    let (mut failed, mut denied) = (0, 0);
    let mut note = |what: String, r: Result<(), NvmlError>| match r {
        Ok(()) => parts.push(what),
        Err(e) => {
            failed += 1;
            denied += usize::from(matches!(e, NvmlError::NoPermission));
            parts.push(format!("{what}: {}", why(&e)));
        }
    };
    if let Some(w) = t.power_limit {
        note(
            format!("power limit {w} W"),
            d.set_power_management_limit(w * 1000),
        );
    }
    if let Some([lo, hi]) = t.lock_core_clocks {
        note(
            format!("core clocks {lo}-{hi} MHz"),
            d.set_gpu_locked_clocks(GpuLockedClocksSetting::Numeric {
                min_clock_mhz: lo,
                max_clock_mhz: hi,
            }),
        );
    }
    if let Some([lo, hi]) = t.lock_mem_clocks {
        note(
            format!("memory clocks {lo}-{hi} MHz"),
            d.set_mem_locked_clocks(lo, hi),
        );
    }
    if let Some(o) = t.core_offset {
        note(format!("core offset {o:+} MHz"), d.set_gpc_clock_vf_offset(o));
    }
    if let Some(o) = t.mem_offset {
        note(format!("memory offset {o:+} MHz"), d.set_mem_clock_vf_offset(o));
    }
    if let Some(f) = t.fan {
        let r = d
            .num_fans()
            .and_then(|n| (0..n).try_for_each(|i| d.set_fan_speed(i, f)));
        note(format!("fan {f}%"), r);
    }
    Applied {
        text: parts.join(", "),
        failed,
        denied,
    }
}

/// Puts the GPU at `pci_bus_id` back to its defaults: clocks unlocked, no
/// offsets, automatic fans, the default power limit.
pub fn reset(nvml: &Nvml, pci_bus_id: &str) -> (String, bool) {
    let mut d = match device(nvml, pci_bus_id) {
        Ok(d) => d,
        Err(e) => return (e, false),
    };
    let mut failed = Vec::new();
    let mut note = |what: &str, r: Result<(), NvmlError>| {
        if let Err(e) = r {
            failed.push(format!("{what}: {}", why(&e)));
        }
    };
    note("core clocks", d.reset_gpu_locked_clocks());
    note("memory clocks", d.reset_mem_locked_clocks());
    note("core offset", d.set_gpc_clock_vf_offset(0));
    note("memory offset", d.set_mem_clock_vf_offset(0));
    let fans = d
        .num_fans()
        .and_then(|n| (0..n).try_for_each(|i| d.set_default_fan_speed(i)));
    note("fans", fans);
    let pl = d
        .power_management_limit_default()
        .and_then(|w| d.set_power_management_limit(w));
    note("power limit", pl);
    if failed.is_empty() {
        ("reset to defaults".into(), true)
    } else {
        (format!("reset, except {}", failed.join(", ")), false)
    }
}

/// The GPU's current settings, one line.
pub fn show(nvml: &Nvml, pci_bus_id: &str) -> String {
    let d = match device(nvml, pci_bus_id) {
        Ok(d) => d,
        Err(e) => return e,
    };
    let na = || "-".to_string();
    let pl = d
        .power_management_limit()
        .map_or_else(|_| na(), |mw| format!("{} W", mw / 1000));
    let range = d.power_management_limit_constraints().map_or_else(
        |_| na(),
        |c| format!("{}-{} W", c.min_limit / 1000, c.max_limit / 1000),
    );
    let co = d
        .gpc_clock_vf_offset()
        .map_or_else(|_| na(), |o| format!("{o:+}"));
    let mo = d
        .mem_clock_vf_offset()
        .map_or_else(|_| na(), |o| format!("{o:+}"));
    let fans = d.num_fans().map_or_else(
        |_| na(),
        |n| {
            (0..n)
                .map(|i| d.fan_speed(i).map_or_else(|_| na(), |f| format!("{f}%")))
                .collect::<Vec<_>>()
                .join("/")
        },
    );
    format!("power limit {pl} (range {range}), core offset {co} MHz, memory offset {mo} MHz, fans {fans}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_and_flags() {
        let cfg: TuneConfig = toml::from_str(
            "power_limit = 150\nlock_core_clocks = [1400, 1600]\n[gpu.3]\npower_limit = 140\nfan = 70\n",
        )
        .unwrap();
        cfg.check().unwrap();
        assert_eq!(cfg.for_device(0).power_limit, Some(150));
        let t3 = cfg.for_device(3);
        assert_eq!(t3.power_limit, Some(140));
        assert_eq!(t3.lock_core_clocks, Some([1400, 1600]));
        assert_eq!(t3.fan, Some(70));
        // A command-line flag wins everywhere, overrides included.
        let flags = Tuning {
            power_limit: Some(170),
            ..Tuning::default()
        };
        let c = cfg.with(&flags);
        assert_eq!(c.for_device(3).power_limit, Some(170));
        assert_eq!(c.for_device(3).fan, Some(70));
        assert!(TuneConfig::default().is_empty());
    }

    #[test]
    fn refuses_bad_values() {
        let bad = |s: &str| toml::from_str::<TuneConfig>(s).unwrap().check().is_err();
        assert!(bad("lock_core_clocks = [1600, 1400]"));
        assert!(bad("fan = 101"));
        assert!(bad("[gpu.x]\nfan = 50"));
        assert!(toml::from_str::<TuneConfig>("power_limt = 150").is_err());
    }
}
