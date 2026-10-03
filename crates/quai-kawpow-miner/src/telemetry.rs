//! GPU telemetry through NVML (loaded at runtime; absent → no readings).

use nvml_wrapper::Nvml;
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};

use crate::stats::DeviceReading;

/// An NVML handle for one device, found by PCI bus id when possible.
pub struct Telemetry {
    nvml: Nvml,
    index: u32,
}

impl Telemetry {
    /// Opens NVML; `pci_bus_id` like `00000000:65:00.0` picks the device,
    /// else `fallback_index`. `None` when NVML is unavailable.
    pub fn open(pci_bus_id: Option<&str>, fallback_index: u32) -> Option<Telemetry> {
        let nvml = Nvml::init().ok()?;
        let index = pci_bus_id
            .and_then(|id| nvml.device_by_pci_bus_id(id).ok())
            .and_then(|d| d.index().ok())
            .unwrap_or(fallback_index);
        nvml.device_by_index(index).ok()?;
        Some(Telemetry { nvml, index })
    }

    /// A reading (fields missing on cards that don't report them).
    pub fn read(&self) -> DeviceReading {
        let Ok(d) = self.nvml.device_by_index(self.index) else {
            return DeviceReading::default();
        };
        DeviceReading {
            temperature_c: d.temperature(TemperatureSensor::Gpu).ok(),
            power_w: d.power_usage().ok().map(|mw| f64::from(mw) / 1000.0),
            power_limit_w: d
                .enforced_power_limit()
                .ok()
                .map(|mw| f64::from(mw) / 1000.0),
            fan_pct: d.fan_speed(0).ok(),
            mem_clock_mhz: d.clock_info(Clock::Memory).ok(),
            sm_clock_mhz: d.clock_info(Clock::SM).ok(),
        }
    }
}
