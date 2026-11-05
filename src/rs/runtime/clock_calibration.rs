//! Paired host/device clock sample and wrap-aware timestamp mapping.
//!
//! Donor: oacpp/source/cpp/include/oa/runtime/clockCalibration.h.
//! Adaptation: Rust naming; device-side sampling lives in
//! `runtime::device` behind `Engine::calibrate_clock`.

/// Host clock paired with the Vulkan device timestamp domain. Host timestamps
/// are nanoseconds in the corresponding POSIX clock domain; device timestamps
/// are ticks and must be scaled by `device_nanoseconds_per_tick`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HostClockDomain {
	#[default]
	Monotonic,
	MonotonicRaw,
}

/// One quasi-simultaneous host/device clock sample.
/// `maximum_deviation_nanoseconds` is the Vulkan implementation's uncertainty
/// bound for the pair, not an OA estimate. Consumers should recalibrate
/// long-running traces periodically.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClockCalibration {
	pub device_timestamp_ticks: u64,
	pub host_timestamp_nanoseconds: u64,
	pub maximum_deviation_nanoseconds: u64,
	pub device_nanoseconds_per_tick: f64,
	pub device_timestamp_valid_bits: u32,
	pub host_clock_domain: HostClockDomain,
}

impl ClockCalibration {
	/// Map a query-pool timestamp into the calibrated host clock.
	///
	/// The signed modular delta handles Vulkan timestamp wrap for queues
	/// exposing fewer than 64 valid bits. Values farther than half a
	/// device-clock period from this sample are inherently ambiguous and
	/// require a newer calibration. Returns 0 when the sample fields cannot
	/// describe a usable mapping.
	pub fn host_nanoseconds_for_device_timestamp(&self, device_timestamp_ticks: u64) -> f64 {
		if self.device_timestamp_valid_bits == 0
			|| self.device_timestamp_valid_bits > 64
			|| self.device_nanoseconds_per_tick <= 0.0
		{
			return 0.0;
		}
		let signed_delta = if self.device_timestamp_valid_bits == 64 {
			device_timestamp_ticks.wrapping_sub(self.device_timestamp_ticks) as i64
		} else {
			let modulus = 1_u64 << self.device_timestamp_valid_bits;
			let mask = modulus - 1;
			let delta = device_timestamp_ticks.wrapping_sub(self.device_timestamp_ticks) & mask;
			if delta >= (modulus >> 1) {
				-((modulus - delta) as i64)
			} else {
				delta as i64
			}
		};
		self.host_timestamp_nanoseconds as f64 + signed_delta as f64 * self.device_nanoseconds_per_tick
	}
}

#[cfg(test)]
#[path = "../../../test/rs/runtime/clock_calibration_unit.rs"]
mod tests;
