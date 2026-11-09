//! One retained device domain and its queried information.

pub(in crate::runtime) mod buffer;
pub(in crate::runtime) mod command;
pub(in crate::runtime) mod commands;
pub(in crate::runtime) mod descriptor;
pub(in crate::runtime) mod dispatch;
pub(in crate::runtime) mod image;
pub(super) mod jit;
pub(in crate::runtime) mod pipeline;
#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "private ML-KEM qualification route; public secret operation admission remains planned"
	)
)]
pub(in crate::runtime) mod pqc;

pub(in crate::runtime) mod retirement;
#[cfg_attr(
	not(test),
	expect(
		dead_code,
		reason = "private PQC storage substrate; secret algorithms are not admitted"
	)
)]
pub(in crate::runtime) mod secret_buffer;
pub(in crate::runtime) mod swapchain;
pub(in crate::runtime) mod timeline;
pub(in crate::runtime) mod timestamp;
pub(in crate::runtime) mod video;
pub(in crate::runtime) mod vkcache;

mod configuration;
pub(in crate::runtime) mod features;
pub(super) mod info;
pub(super) mod logical;
pub(in crate::runtime) mod physical;

use std::sync::Arc;

/// Bitwise membership test for generated `vk::*FlagBits*` newtypes.
///
/// The generated bindings expose `BitAnd`/`BitOr` but no named membership
/// helper; this restores the conventional `flags.contains(bit)` spelling for
/// private native code.
pub(in crate::runtime) trait NativeFlagBits:
	Copy + PartialEq + std::ops::BitAnd<Output = Self>
{
	fn contains(self, member: Self) -> bool {
		(self & member) == member
	}
}

macro_rules! impl_native_flag_bits {
	($($ty:ty),* $(,)?) => {
		$(impl NativeFlagBits for $ty {})*
	};
}

impl_native_flag_bits! {
	vk::FormatFeatureFlagBits,
	vk::ImageCreateFlagBits,
	vk::ImageUsageFlagBits,
	vk::QueueFlagBits,
	vk::ShaderStageFlagBits,
	vk::CompositeAlphaFlagBitsKHR,
	vk::SampleCountFlagBits,
	vk::VideoCapabilityFlagBitsKHR,
	vk::VideoChromaSubsamplingFlagBitsKHR,
	vk::VideoCodingControlFlagBitsKHR,
	vk::VideoComponentBitDepthFlagBitsKHR,
	vk::VideoDecodeCapabilityFlagBitsKHR,
	vk::VideoDecodeH264PictureLayoutFlagBitsKHR,
	vk::VideoEncodeFeedbackFlagBitsKHR,
	vk::VideoEncodeH265CtbSizeFlagBitsKHR,
	vk::VideoEncodeH265TransformBlockSizeFlagBitsKHR,
	vk::VideoEncodeRateControlModeFlagBitsKHR,
	vk::AccessFlagBits2,
	vk::PipelineStageFlagBits2,
	vk::BufferUsageFlagBits,
	vk::QueryResultFlagBits,
}

use configuration::DeviceConfiguration;
use logical::DeviceLogical;
pub(in crate::runtime) use physical::DevicePhysical;

/// Engine-private execution domain, retained by resources through `Arc<Device>`.
/// Queried facts and executor resources each have one authoritative owner.
/// Owned executor fields never retain this Arc, preventing ownership cycles.
pub(in crate::runtime) struct Device {
	// Drop executor resources before releasing their owning Vulkan instance.
	logical: DeviceLogical,
	physical: DevicePhysical,
	configuration: DeviceConfiguration,
	instance: super::instance::Instance,
}

impl Device {
	pub(in crate::runtime) fn new(
		instance: &super::instance::Instance,
		physical: DevicePhysical,
	) -> crate::Result<Arc<Self>> {
		let configuration = DeviceConfiguration::new(instance, &physical);
		let logical = DeviceLogical::new(instance, &physical, &configuration)?;
		Ok(Arc::new(Self {
			logical,
			physical,
			configuration,
			instance: instance.clone(),
		}))
	}

	pub(in crate::runtime) fn logical(&self) -> &DeviceLogical {
		&self.logical
	}

	pub(in crate::runtime) fn compute_shared_memory_bytes(&self) -> u32 {
		self.physical.limits.max_compute_shared_memory_size
	}

	/// Queried per-dimension compute dispatch group limits.
	pub(in crate::runtime) const fn max_compute_work_group_count(&self) -> [u32; 3] {
		self.physical.limits.max_compute_work_group_count
	}

	/// Flatten linear work across two queried dispatch dimensions.
	pub(in crate::runtime) fn linear_dispatch_grid(
		&self,
		count: u32,
		group_size: u32,
	) -> crate::Result<[u32; 3]> {
		let limits = self.physical.limits;
		if group_size == 0
			|| group_size > limits.max_compute_work_group_size[0]
			|| group_size > limits.max_compute_work_group_invocations
		{
			return Err(crate::Error::invalid_argument(
				"linear dispatch group size exceeds device limits",
			));
		}
		let groups = count.div_ceil(group_size).max(1);
		let x = groups.min(limits.max_compute_work_group_count[0]);
		if x == 0 || limits.max_compute_work_group_count[1] == 0 {
			return Err(crate::Error::missing_capability(
				"device has no usable compute dispatch grid",
			));
		}
		let y = groups.div_ceil(x);
		if y > limits.max_compute_work_group_count[1] {
			return Err(crate::Error::invalid_argument(
				"linear work exceeds queried compute dispatch grid",
			));
		}
		Ok([x, y, 1])
	}

	/// Return whether the selected device can calibrate its timestamp domain
	/// against a POSIX monotonic host clock.
	pub(in crate::runtime) fn supports_clock_calibration(&self) -> bool {
		self.physical.limits.compute_timestamp_valid_bits != 0
			&& (self
				.configuration
				.extension_enabled(vk::KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)
				|| self
					.configuration
					.extension_enabled(vk::EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME))
	}

	/// Sample paired device/host timestamps and retain the lowest-deviation
	/// pair, matching the donor `oa::Engine::calibrateClock`.
	pub(in crate::runtime) fn calibrate_clock(
		&self,
		sample_count: u32,
	) -> crate::Result<crate::ClockCalibration> {
		if !(1..=1024).contains(&sample_count) {
			return Err(crate::Error::invalid_argument(
				"clock calibration sample count must be in [1, 1024]",
			));
		}
		if !self.supports_clock_calibration() {
			return Err(crate::Error::unavailable(
				"selected device does not expose calibrated timestamps",
			));
		}
		let use_khr = self
			.configuration
			.extension_enabled(vk::KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
		let instance = self.instance.raw();
		let query_domains = |count: *mut u32, domains: *mut vk::TimeDomainKHR| unsafe {
			if use_khr {
				instance.get_physical_device_calibrateable_time_domains_khr(
					self.physical.handle,
					count,
					domains,
				)
			} else {
				instance.get_physical_device_calibrateable_time_domains_ext(
					self.physical.handle,
					count,
					domains,
				)
			}
		};
		let mut domain_count = 0_u32;
		let count_result = query_domains(&mut domain_count, std::ptr::null_mut());
		match count_result {
			Ok(vk::Result::SUCCESS) if domain_count > 0 => {}
			Err(vk::CallError::MissingCommand(_)) => {
				return Err(crate::Error::unavailable(
					"calibrated timestamp entry points are unavailable",
				));
			}
			_ => {
				return Err(crate::Error::backend_dispatch_failure(
					"Vulkan",
					"calibrated timestamp domain query",
					count_result,
				));
			}
		}
		let mut domains = vec![vk::TimeDomainKHR::DEVICE_KHR; domain_count as usize];
		let list_result = query_domains(&mut domain_count, domains.as_mut_ptr());
		if let Err(vk::CallError::MissingCommand(_)) = list_result {
			return Err(crate::Error::unavailable(
				"calibrated timestamp entry points are unavailable",
			));
		}
		if list_result != Ok(vk::Result::SUCCESS) {
			return Err(crate::Error::backend_dispatch_failure(
				"Vulkan",
				"calibrated timestamp domain enumeration",
				list_result,
			));
		}
		let has_device = domains.contains(&vk::TimeDomainKHR::DEVICE_KHR);
		let has_monotonic = domains.contains(&vk::TimeDomainKHR::CLOCK_MONOTONIC_KHR);
		let has_monotonic_raw = domains.contains(&vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW_KHR);
		if !has_device || !(has_monotonic || has_monotonic_raw) {
			return Err(crate::Error::unavailable(
				"device and POSIX monotonic clock domains cannot be calibrated together",
			));
		}

		let host_domain = if has_monotonic_raw {
			vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW_KHR
		} else {
			vk::TimeDomainKHR::CLOCK_MONOTONIC_KHR
		};
		let infos = [
			vk::CalibratedTimestampInfoKHR::new().set_time_domain(vk::TimeDomainKHR::DEVICE_KHR),
			vk::CalibratedTimestampInfoKHR::new().set_time_domain(host_domain),
		];
		let device = self.raw();
		let mut best = crate::ClockCalibration::default();
		let mut have_best = false;
		for _ in 0..sample_count {
			let mut timestamps = [0_u64; 2];
			let mut maximum_deviation = 0_u64;
			// SAFETY: infos describes two domains; timestamps has room for two
			// results and maximum_deviation is a live u64. The extension is
			// enabled on this device.
			let sample_result = unsafe {
				if use_khr {
					device.get_calibrated_timestamps_khr(
						2,
						infos.as_ptr(),
						timestamps.as_mut_ptr(),
						&mut maximum_deviation,
					)
				} else {
					device.get_calibrated_timestamps_ext(
						2,
						infos.as_ptr(),
						timestamps.as_mut_ptr(),
						&mut maximum_deviation,
					)
				}
			};
			if let Err(vk::CallError::MissingCommand(_)) = sample_result {
				return Err(crate::Error::unavailable(
					"calibrated timestamp entry points are unavailable",
				));
			}
			if sample_result != Ok(vk::Result::SUCCESS) {
				return Err(crate::Error::backend_dispatch_failure(
					"Vulkan",
					"calibrated timestamp sample",
					sample_result,
				));
			}
			if have_best && maximum_deviation >= best.maximum_deviation_nanoseconds {
				continue;
			}
			let valid_bits = self.physical.limits.compute_timestamp_valid_bits;
			if valid_bits < 64 {
				timestamps[0] &= (1_u64 << valid_bits) - 1;
			}
			best = crate::ClockCalibration {
				device_timestamp_ticks: timestamps[0],
				host_timestamp_nanoseconds: timestamps[1],
				maximum_deviation_nanoseconds: maximum_deviation,
				device_nanoseconds_per_tick: self.physical.limits.timestamp_period_ns,
				device_timestamp_valid_bits: valid_bits,
				host_clock_domain: if has_monotonic_raw {
					crate::HostClockDomain::MonotonicRaw
				} else {
					crate::HostClockDomain::Monotonic
				},
			};
			have_best = true;
		}
		Ok(best)
	}
}

pub use info::{DeviceKind, DeviceSummary};
