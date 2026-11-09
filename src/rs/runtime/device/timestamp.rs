use std::{sync::Arc, time::Duration};

use crate::{Error, Result};

use crate::runtime::Device;

const QUERY_COUNT: u32 = 2;
const START_QUERY: u32 = 0;
const END_QUERY: u32 = 1;

/// Shared ownership of one immutable timestamp result pair.
#[derive(Clone)]
pub(crate) struct TimestampPair {
	inner: Arc<TimestampPairInner>,
}

struct TimestampPairInner {
	device: Arc<Device>,
	pool: vk::QueryPool,
	period_ns: f64,
	valid_bits: u32,
}

impl TimestampPair {
	pub(super) fn new(device: &Arc<Device>, period_ns: f64, valid_bits: u32) -> Result<Self> {
		if valid_bits == 0 || valid_bits > u64::BITS || !period_ns.is_finite() || period_ns <= 0.0 {
			return Err(Error::missing_capability(
				"selected Vulkan compute queue does not support device timestamps",
			));
		}
		let create_info = vk::QueryPoolCreateInfo::new()
			.set_query_type(vk::QueryType::TIMESTAMP)
			.set_query_count(QUERY_COUNT);
		// SAFETY: `device` is live, the timestamp query count is nonzero, and no
		// allocation callbacks are installed. The returned pool is owned by the
		// timestamp pair and destroyed through the same device.
		let pool = unsafe { device.raw().create_query_pool(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "timestamp-pool creation", source))?;

		Ok(Self {
			inner: Arc::new(TimestampPairInner {
				device: device.clone(),
				pool,
				period_ns,
				valid_bits,
			}),
		})
	}

	pub(super) fn record_begin(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
		// SAFETY: the command buffer is recording outside a render/video scope and
		// belongs to the same device and compute-capable queue family as this pool.
		// The fresh pool's two queries are first reset to unavailable, then query zero
		// is written at TOP_OF_PIPE, a single stage valid for a compute queue.
		unsafe {
			self.inner.device.raw().cmd_reset_query_pool(
				command_buffer,
				self.inner.pool,
				0,
				QUERY_COUNT,
			)?;
			self.inner.device.sync_commands().write_timestamp(
				self.inner.device.raw(),
				command_buffer,
				vk::PipelineStageFlagBits2::TOP_OF_PIPE_BIT,
				self.inner.pool,
				START_QUERY,
			)?;
		}
		Ok(())
	}

	pub(super) fn record_end(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
		// SAFETY: the command buffer is still recording on the pool's compute queue;
		// query one is unavailable and BOTTOM_OF_PIPE is one queue-valid stage after
		// the recorded graph region.
		unsafe {
			self.inner.device.sync_commands().write_timestamp(
				self.inner.device.raw(),
				command_buffer,
				vk::PipelineStageFlagBits2::BOTTOM_OF_PIPE_BIT,
				self.inner.pool,
				END_QUERY,
			)?;
		}
		Ok(())
	}

	/// Return whether this pair was created on `device`.
	pub(in crate::runtime) fn belongs_to(&self, device: &Arc<Device>) -> bool {
		self.inner.device.same_as(device)
	}

	pub(in crate::runtime) fn duration(&self) -> Result<Duration> {
		let mut results = [0_u64; QUERY_COUNT as usize];
		// SAFETY: the result slice has one aligned u64 per query. The pool remains
		// live through `self`; callers establish completion before this non-waiting
		// read, and no query is ever reset or reused.
		match unsafe {
			self.inner.device.raw().get_query_pool_results(
				self.inner.pool,
				0,
				QUERY_COUNT,
				std::slice::from_raw_parts_mut(
					results.as_mut_ptr() as *mut u8,
					results.len() * std::mem::size_of::<u64>(),
				),
				std::mem::size_of::<u64>() as vk::DeviceSize,
				vk::QueryResultFlagBits::_64_BIT,
			)
		} {
			Ok(vk::Result::SUCCESS) => duration_from_ticks(
				results[START_QUERY as usize],
				results[END_QUERY as usize],
				self.inner.valid_bits,
				self.inner.period_ns,
			),
			Ok(vk::Result::NOT_READY) => Err(Error::not_ready("device timestamp results are not ready")),
			result => Err(Error::backend_dispatch_failure(
				"Vulkan",
				"timestamp readback",
				result,
			)),
		}
	}
}

impl Drop for TimestampPairInner {
	fn drop(&mut self) {
		// SAFETY: the final Arc uniquely owns this pool. Command retirement and Event
		// ownership retain an Arc until the submitted timestamp commands complete,
		// and the embedded Arc<Device> keeps the logical device alive through destruction.
		let _ = unsafe { self.device.raw().destroy_query_pool(self.pool, None) };
	}
}

/// The route a fused dispatch executed, for route-observation timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObservedRoute {
	/// The canonical fused tape interpreter ran.
	Tape,
	/// An admitted specialization ran.
	Specialized,
}

/// Per-dispatch timestamps for route observation: one pool with two queries
/// per instrumented node, recorded around each fused dispatch. The pool is
/// shared between command retention (which keeps it alive until the recorded
/// work completes) and the device's jit table (which reads results lazily and
/// never blocks a submission).
pub(crate) struct RouteTimings {
	device: Arc<vk::Device>,
	sync_commands: super::commands::SyncCommands,
	pool: vk::QueryPool,
	period_ns: f64,
	valid_bits: u32,
	/// `(signature, route)` in query-slot order; slot `i` owns queries
	/// `2i` (before) and `2i + 1` (after).
	entries: Vec<(u64, ObservedRoute)>,
}

impl RouteTimings {
	/// Maximum fused dispatches instrumented in one command buffer.
	pub(crate) const MAX_SLOTS: usize = 8;

	pub(crate) fn new(
		device: &Arc<Device>,
		entries: Vec<(u64, ObservedRoute)>,
		period_ns: f64,
		valid_bits: u32,
	) -> Result<Arc<Self>> {
		if entries.is_empty()
			|| entries.len() > Self::MAX_SLOTS
			|| valid_bits == 0
			|| valid_bits > u64::BITS
			|| !period_ns.is_finite()
			|| period_ns <= 0.0
		{
			return Err(Error::invalid_argument(
				"route-observation timings require 1..=MAX_SLOTS entries and a timestamp-capable queue",
			));
		}
		let create_info = vk::QueryPoolCreateInfo::new()
			.set_query_type(vk::QueryType::TIMESTAMP)
			.set_query_count(2 * entries.len() as u32);
		// SAFETY: `device` is live, the query count is nonzero, and no allocation
		// callbacks are installed. The pool is destroyed through the same device.
		let pool = unsafe { device.raw().create_query_pool(&create_info, None) }
			.map_err(|source| Error::backend_failure("Vulkan", "route-timing pool creation", source))?;
		Ok(Arc::new(Self {
			device: device.raw_arc(),
			sync_commands: device.sync_commands().clone(),
			pool,
			period_ns,
			valid_bits,
			entries,
		}))
	}

	/// Reset every query; recorded once before the first instrumented slot.
	pub(super) fn record_reset(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
		// SAFETY: the command buffer is recording on the pool's queue family and
		// this recording exclusively owns the pool's query range.
		unsafe {
			self.device.cmd_reset_query_pool(
				command_buffer,
				self.pool,
				0,
				2 * self.entries.len() as u32,
			)?;
		}
		Ok(())
	}

	/// Timestamp the start of instrumented `slot`.
	pub(super) fn record_before(&self, command_buffer: vk::CommandBuffer, slot: usize) -> Result<()> {
		self.write(
			command_buffer,
			vk::PipelineStageFlagBits2::TOP_OF_PIPE_BIT,
			2 * slot as u32,
		)
	}

	/// Timestamp the end of instrumented `slot`.
	pub(super) fn record_after(&self, command_buffer: vk::CommandBuffer, slot: usize) -> Result<()> {
		self.write(
			command_buffer,
			vk::PipelineStageFlagBits2::BOTTOM_OF_PIPE_BIT,
			2 * slot as u32 + 1,
		)
	}

	fn write(
		&self,
		command_buffer: vk::CommandBuffer,
		stage: vk::PipelineStageFlagBits2,
		query: u32,
	) -> Result<()> {
		// SAFETY: the command buffer is recording on the pool's queue family;
		// `query` is inside this pool's reset range.
		unsafe {
			self
				.sync_commands
				.write_timestamp(&self.device, command_buffer, stage, self.pool, query)?;
		}
		Ok(())
	}

	/// Read completed durations without waiting. `None` means the recorded
	/// submission has not finished writing every query yet (or never ran).
	pub(in crate::runtime) fn try_durations(&self) -> Option<Vec<(u64, ObservedRoute, Duration)>> {
		let count = 2 * self.entries.len();
		let mut results = vec![0_u64; count];
		// SAFETY: one aligned u64 per query. The pool stays live through `self`;
		// no WAIT flag, so an unfinished submission reports NOT_READY instead of
		// blocking. No query is ever reset or reused by this path.
		match unsafe {
			self.device.get_query_pool_results(
				self.pool,
				0,
				2 * self.entries.len() as u32,
				std::slice::from_raw_parts_mut(
					results.as_mut_ptr() as *mut u8,
					results.len() * std::mem::size_of::<u64>(),
				),
				std::mem::size_of::<u64>() as vk::DeviceSize,
				vk::QueryResultFlagBits::_64_BIT,
			)
		} {
			Ok(vk::Result::SUCCESS) => Some(
				self
					.entries
					.iter()
					.enumerate()
					.filter_map(|(slot, (signature, route))| {
						duration_from_ticks(
							results[2 * slot],
							results[2 * slot + 1],
							self.valid_bits,
							self.period_ns,
						)
						.ok()
						.map(|duration| (*signature, *route, duration))
					})
					.collect(),
			),
			_ => None,
		}
	}
}

impl Drop for RouteTimings {
	fn drop(&mut self) {
		// SAFETY: command retention keeps the pool live through completion and
		// retains the Device until retirement. The jit table owns only
		// the native-device Arc, avoiding Device -> jit -> timings -> Device.
		// DeviceLogical drops that table before destroying the native device.
		let _ = unsafe { self.device.destroy_query_pool(self.pool, None) };
	}
}

fn duration_from_ticks(start: u64, end: u64, valid_bits: u32, period_ns: f64) -> Result<Duration> {
	let mut ticks = end.wrapping_sub(start);
	if valid_bits < u64::BITS {
		ticks &= (1_u64 << valid_bits) - 1;
	}
	Duration::try_from_secs_f64((ticks as f64 * period_ns) / 1_000_000_000.0)
		.map_err(|source| Error::backend_failure("Vulkan", "timestamp conversion", source))
}
