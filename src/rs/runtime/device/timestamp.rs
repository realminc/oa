use std::{sync::Arc, time::Duration};

use crate::{Error, Result};

use crate::runtime::Device;

const QUERY_COUNT: u32 = 2;
const START_QUERY: u32 = 0;
const END_QUERY: u32 = 1;

/// Shared ownership of one immutable timestamp result pair.
#[derive(Clone)]
pub(in crate::runtime) struct TimestampPair {
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
		let create_info = vk::QueryPoolCreateInfo::default()
			.query_type(vk::QueryType::TIMESTAMP)
			.query_count(QUERY_COUNT);
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

fn duration_from_ticks(start: u64, end: u64, valid_bits: u32, period_ns: f64) -> Result<Duration> {
	let mut ticks = end.wrapping_sub(start);
	if valid_bits < u64::BITS {
		ticks &= (1_u64 << valid_bits) - 1;
	}
	Duration::try_from_secs_f64((ticks as f64 * period_ns) / 1_000_000_000.0)
		.map_err(|source| Error::backend_failure("Vulkan", "timestamp conversion", source))
}
