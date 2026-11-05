//! Named host/device measurement regions.
//!
//! Donor: `oacpp/source/cpp/include/oa/runtime/timer.h`,
//! `timer.cpp`, and `timerRegistry.{h,cpp}`.
//! Adaptation: Rust naming (`Timer`, `TimerDomain`); each device timer owns one
//! dedicated timestamp pair instead of a pooled registry slot — the donor's
//! slot pooling is an internal allocation optimization with the same public
//! contract. Device regions are recorded by
//! [`Engine::checkpoint_with_timer`]; [`Timer::commit`] never waits and
//! requires the attached submission event to be complete.

use std::sync::Arc;

use crate::{Error, Result, Timestamp};

use super::{Device, Engine, Event, TimestampPair};

/// Clock domain measured by one [`Timer`]. `Auto` resolves to `Device` on
/// [`Timer::init`] and to `Host` on the first [`Timer::begin`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimerDomain {
	/// Resolved by the first timing operation.
	#[default]
	Auto,
	/// Host-side monotonic region bracketed by [`Timer::begin`]/[`Timer::end`].
	Host,
	/// Device-side region bracketed by one timer-tagged engine submission.
	Device,
}

/// One named measurement region.
///
/// A host timer brackets arbitrary host code. A device timer must be
/// initialized against an engine, then one
/// [`Engine::checkpoint_with_timer`] records the device timestamps around the
/// submitted batch. The device path is single-flight: a timer cannot tag a
/// new submission while its previous one is still pending.
pub struct Timer {
	domain: TimerDomain,
	name: String,
	host_start: Option<Timestamp>,
	host_end: Option<Timestamp>,
	host_running: bool,
	host_complete: bool,
	device_pair: Option<TimestampPair>,
	device_event: Option<Event>,
	last_milliseconds: f64,
	last_units: f64,
}

impl Timer {
	/// Create a timer in `domain` named `name`.
	pub fn new(domain: TimerDomain, name: impl Into<String>) -> Self {
		Self {
			domain,
			name: name.into(),
			host_start: None,
			host_end: None,
			host_running: false,
			host_complete: false,
			device_pair: None,
			device_event: None,
			last_milliseconds: 0.0,
			last_units: 1.0,
		}
	}

	/// Resolve `Auto` to `Device` and acquire this engine's device timestamp
	/// backend. A host timer needs no engine and initialization is a no-op.
	///
	/// # Errors
	///
	/// Returns an error when the domain is `Device` and the engine's compute
	/// queue cannot provide timestamps.
	pub fn init(&mut self, engine: &Engine, name: impl Into<String>) -> Result<()> {
		let name = name.into();
		if !name.is_empty() {
			self.name = name;
		}
		if self.domain == TimerDomain::Auto {
			self.domain = TimerDomain::Device;
		}
		if self.domain == TimerDomain::Host {
			return Ok(());
		}
		self.device_pair = Some(engine.device_timestamp_pair()?);
		self.device_event = None;
		Ok(())
	}

	/// Begin a host-side region. `Auto` resolves to `Host` here.
	///
	/// # Errors
	///
	/// Returns an error when the timer resolves to `Device` — device regions are
	/// recorded by [`Engine::checkpoint_with_timer`] — or when a previous host
	/// region is still running.
	pub fn begin(&mut self) -> Result<()> {
		if self.domain == TimerDomain::Auto {
			self.domain = TimerDomain::Host;
		}
		if self.domain != TimerDomain::Host {
			return Err(Error::failed_precondition(
				"device timer regions are recorded by Engine::checkpoint_with_timer",
			));
		}
		if self.host_running {
			return Err(Error::failed_precondition(
				"host timer begin requires the previous region to end",
			));
		}
		self.host_start = Some(Timestamp::now());
		self.host_running = true;
		self.host_complete = false;
		Ok(())
	}

	/// End the running host-side region.
	///
	/// # Errors
	///
	/// Returns an error without a matching [`Timer::begin`].
	pub fn end(&mut self) -> Result<()> {
		if self.domain != TimerDomain::Host || !self.host_running {
			return Err(Error::failed_precondition(
				"host timer end requires a matching begin",
			));
		}
		self.host_end = Some(Timestamp::now());
		self.host_running = false;
		self.host_complete = true;
		Ok(())
	}

	/// Commit the measured region in milliseconds and remember `units` for
	/// [`Timer::throughput`]. Never waits: a device timer requires its tagged
	/// submission to be complete.
	///
	/// # Errors
	///
	/// - `FailedPrecondition`: no completed region exists (host timer without a
	///   finished `begin`/`end`, device timer without a tagged submission).
	/// - `NotReady`: the device submission is still pending.
	/// - Vulkan timestamp readback failures.
	pub fn commit(&mut self, units: f64) -> Result<f64> {
		self.last_milliseconds = match self.domain {
			TimerDomain::Device => {
				let pair = self.device_pair.as_ref().ok_or_else(|| {
					Error::failed_precondition("device timer commit requires init and a submitted region")
				})?;
				let event = self.device_event.as_ref().ok_or_else(|| {
					Error::failed_precondition("device timer commit requires a submitted region")
				})?;
				if !event.is_complete()? {
					return Err(Error::not_ready("device timer submission is still pending"));
				}
				pair.duration()?.as_secs_f64() * 1e3
			}
			TimerDomain::Host => {
				if !self.host_complete {
					return Err(Error::failed_precondition(
						"host timer commit requires a completed begin/end region",
					));
				}
				self.host_complete = false;
				(self.host_end.expect("complete region has an end")
					- self.host_start.expect("complete region has a start"))
				.to_ms()
			}
			TimerDomain::Auto => {
				return Err(Error::failed_precondition("timer has no measured region"));
			}
		};
		self.last_units = units;
		Ok(self.last_milliseconds)
	}

	/// Return the last committed region duration in milliseconds.
	pub const fn elapsed_milliseconds(&self) -> f64 {
		self.last_milliseconds
	}

	/// Return committed units per second, or `0.0` before a nonzero commit.
	pub const fn throughput(&self) -> f64 {
		if self.last_milliseconds <= 0.0 {
			return 0.0;
		}
		self.last_units / (self.last_milliseconds / 1000.0)
	}

	/// Return the resolved or configured domain.
	pub const fn domain(&self) -> TimerDomain {
		self.domain
	}

	/// Return whether this timer measures host regions.
	pub const fn is_host(&self) -> bool {
		matches!(self.domain, TimerDomain::Host)
	}

	/// Return whether this timer measures device regions.
	pub const fn is_device(&self) -> bool {
		matches!(self.domain, TimerDomain::Device)
	}

	/// Return whether this timer can measure in its current domain. A host
	/// timer is always initialized; a device timer requires [`Timer::init`].
	pub fn is_initialized(&self) -> bool {
		match self.domain {
			TimerDomain::Host => true,
			TimerDomain::Device => self.device_pair.is_some(),
			TimerDomain::Auto => false,
		}
	}

	/// Return whether a measurement region is in flight: a running host region,
	/// or a device submission that has not completed.
	pub fn is_pending(&self) -> bool {
		match self.domain {
			TimerDomain::Host => self.host_running,
			TimerDomain::Device => self
				.device_event
				.as_ref()
				.is_some_and(|event| !event.is_complete().unwrap_or(true)),
			TimerDomain::Auto => false,
		}
	}

	/// Return the configured timer name.
	pub fn name(&self) -> &str {
		&self.name
	}

	/// Validate a timer-tagged submission against this timer's device state.
	pub(in crate::runtime) fn prepare_device_submit(&mut self, device: &Arc<Device>) -> Result<()> {
		if self.domain != TimerDomain::Device {
			return Err(Error::failed_precondition(
				"engine submission requires an initialized device timer",
			));
		}
		let pair = self.device_pair.as_ref().ok_or_else(|| {
			Error::failed_precondition("device timer requires init before timed submission")
		})?;
		if !pair.belongs_to(device) {
			return Err(Error::invalid_argument(
				"device timer was initialized by another engine",
			));
		}
		if self.is_pending() {
			return Err(Error::failed_precondition(
				"device timer is still pending; commit the previous region before resubmitting",
			));
		}
		Ok(())
	}

	pub(in crate::runtime) fn device_pair(&self) -> Option<&TimestampPair> {
		self.device_pair.as_ref()
	}

	pub(in crate::runtime) fn attach_event(&mut self, event: Event) {
		self.device_event = Some(event);
	}
}

#[cfg(test)]
#[path = "../../../test/rs/runtime/timer_unit.rs"]
mod tests;
