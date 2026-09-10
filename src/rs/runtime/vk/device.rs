use std::{
	ffi::CStr,
	mem::ManuallyDrop,
	sync::{Arc, Mutex},
};

use crate::{
	Error, Result,
	runtime::{executable_graph::ExecutableGraph, shader::KernelId},
};

use super::{
	Instance, PhysicalDevice,
	buffer::RecycledBuffer,
	command::{CommandPool, CommandPoolKind, RecordedCommandBuffer, ReusableCommandBuffer},
	descriptor::DescriptorHeap,
	pipeline::ComputePipeline,
	queue::Queue,
	timeline::Timeline,
	timestamp::TimestampPair,
};

const MAX_RECYCLED_STORAGE_BUFFERS: usize = 256;
const MAX_RECYCLED_STORAGE_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone)]
pub(in crate::runtime) struct Device {
	inner: Arc<DeviceInner>,
}

struct DeviceInner {
	handle: ash::Device,
	allocator: ManuallyDrop<vk_mem::Allocator>,
	command_pool: Mutex<CommandPool>,
	video_decode_command_pool: Mutex<Option<CommandPool>>,
	timeline: Timeline,
	descriptors: DescriptorHeap,
	storage_pool: Mutex<StoragePool>,
	pipelines: Vec<ComputePipeline>,
	physical: PhysicalDevice,
	compute_queue: Queue,
	video_decode_queue: Option<Queue>,
	video_encode_queue: Option<Queue>,
	timestamp_period_ns: f64,
	compute_timestamp_valid_bits: u32,
	_instance: Instance,
}

#[derive(Default)]
struct StoragePool {
	free: Vec<RecycledBuffer>,
	bytes: usize,
}

impl Device {
	pub(in crate::runtime) fn video_device_capabilities(
		&self,
	) -> Result<crate::video::VideoDeviceCapabilities> {
		let video = self.inner.physical.video;

		Ok(crate::video::VideoDeviceCapabilities {
			decode_queue_family: video.decode_queue_family,
			encode_queue_family: video.encode_queue_family,
			decode_result_status_queries: video.decode_result_status_queries,
			encode_result_status_queries: video.encode_result_status_queries,
			h264_decode: video.h264_decode,
			h265_decode: video.h265_decode,
			av1_decode: video.av1_decode,
			vp9_decode: video.vp9_decode,
			h264_encode: video.h264_encode,
			h265_encode: video.h265_encode,
			av1_encode: video.av1_encode,
			video_queues_enabled: self.inner.video_decode_queue.is_some()
				|| self.inner.video_encode_queue.is_some(),
			decoder_sessions_available: false,
			encoder_sessions_available: false,
		})
	}

	pub(in crate::runtime) fn video_decode_capabilities(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeCapabilities> {
		super::video::query_decode_capabilities(
			&self.inner._instance,
			&self.inner.physical,
			profile,
		)
	}

	pub(in crate::runtime) fn video_decode_formats(
		&self,
		profile: crate::video::VideoDecodeProfile,
	) -> Result<crate::video::VideoDecodeFormats> {
		super::video::query_decode_formats(&self.inner._instance, &self.inner.physical, profile)
	}

	pub(in crate::runtime) fn new(instance: &Instance, physical: PhysicalDevice) -> Result<Self> {
		let priorities = [1.0_f32];
		let mut queue_family_indices = vec![physical.compute_queue_family];
		for family in [
			physical.video.decode_queue_family,
			physical.video.encode_queue_family,
		]
		.into_iter()
		.flatten()
		{
			if !queue_family_indices.contains(&family) {
				queue_family_indices.push(family);
			}
		}
		let queue_create_infos: Vec<_> = queue_family_indices
			.iter()
			.map(|family| {
				ash::vk::DeviceQueueCreateInfo::default()
					.queue_family_index(*family)
					.queue_priorities(&priorities)
			})
			.collect();
		let mut extensions = Vec::<&CStr>::new();
		if physical.video.decode_queue_family.is_some()
			|| physical.video.encode_queue_family.is_some()
		{
			extensions.push(ash::khr::video_queue::NAME);
		}
		if physical.video.decode_queue_family.is_some() {
			extensions.push(ash::khr::video_decode_queue::NAME);
		}
		if physical.video.h264_decode {
			extensions.push(ash::khr::video_decode_h264::NAME);
		}
		if physical.video.h265_decode {
			extensions.push(ash::khr::video_decode_h265::NAME);
		}
		if physical.video.av1_decode {
			extensions.push(ash::khr::video_decode_av1::NAME);
		}
		if physical.video.vp9_decode {
			extensions.push(c"VK_KHR_video_decode_vp9");
		}
		if physical.video.encode_queue_family.is_some() {
			extensions.push(ash::khr::video_encode_queue::NAME);
		}
		if physical.video.h264_encode {
			extensions.push(ash::khr::video_encode_h264::NAME);
		}
		if physical.video.h265_encode {
			extensions.push(ash::khr::video_encode_h265::NAME);
		}
		if physical.video.av1_encode {
			extensions.push(c"VK_KHR_video_encode_av1");
		}
		let extension_names: Vec<_> = extensions.iter().map(|name| name.as_ptr()).collect();
		let mut features12 = ash::vk::PhysicalDeviceVulkan12Features::default()
			.timeline_semaphore(physical.features.timeline_semaphore)
			.runtime_descriptor_array(physical.features.runtime_descriptor_array)
			.descriptor_binding_partially_bound(
				physical.features.descriptor_binding_partially_bound,
			)
			.descriptor_binding_storage_buffer_update_after_bind(
				physical
					.features
					.descriptor_binding_storage_buffer_update_after_bind,
			)
			.descriptor_binding_update_unused_while_pending(
				physical
					.features
					.descriptor_binding_update_unused_while_pending,
			);
		let mut features13 = ash::vk::PhysicalDeviceVulkan13Features::default()
			.synchronization2(physical.features.synchronization2);
		let create_info = ash::vk::DeviceCreateInfo::default()
			.queue_create_infos(&queue_create_infos)
			.enabled_extension_names(&extension_names)
			.push_next(&mut features12)
			.push_next(&mut features13);

		// SAFETY: the physical device and queue-family index were queried from this
		// instance, one queue exists in the selected family, priorities are in [0, 1],
		// every enabled feature was reported by the chained feature query, and all
		// create-info references remain alive for the duration of the call.
		let handle = unsafe {
			instance
				.raw()
				.create_device(physical.handle, &create_info, None)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "logical-device creation", source))?;

		// SAFETY: logical-device creation requested queue zero from this family, and
		// the logical device remains alive for the returned queue's full lifetime.
		let compute_queue_handle =
			unsafe { handle.get_device_queue(physical.compute_queue_family, 0) };
		let compute_queue = Queue {
			handle: compute_queue_handle,
		};
		let video_decode_queue = physical.video.decode_queue_family.map(|family| Queue {
			// SAFETY: logical-device creation requested queue zero from every unique
			// selected video queue family and the device owns the returned handle.
			handle: unsafe { handle.get_device_queue(family, 0) },
		});
		let video_encode_queue = physical.video.encode_queue_family.map(|family| Queue {
			// SAFETY: same queue-creation proof as the decode queue above.
			handle: unsafe { handle.get_device_queue(family, 0) },
		});
		let mut command_pool = match CommandPool::new(
			&handle,
			physical.compute_queue_family,
			CommandPoolKind::Compute,
		) {
			Ok(command_pool) => command_pool,
			Err(error) => {
				// SAFETY: command-pool creation failed, so the device has no live child
				// resources. Creation used no allocation callbacks.
				unsafe {
					handle.destroy_device(None);
				}
				return Err(error);
			}
		};
		let mut timeline = match Timeline::new(&handle) {
			Ok(timeline) => timeline,
			Err(error) => {
				command_pool.destroy(&handle);
				// SAFETY: timeline creation failed and the command pool was destroyed, so the
				// device has no live children. No callbacks were installed.
				unsafe {
					handle.destroy_device(None);
				}
				return Err(error);
			}
		};

		let mut allocator_create_info =
			vk_mem::AllocatorCreateInfo::new(instance.raw(), &handle, physical.handle);
		allocator_create_info.vulkan_api_version = ash::vk::API_VERSION_1_3;

		// SAFETY: the instance, physical device, and logical device remain valid in
		// `DeviceInner` until after the allocator is explicitly destroyed.
		let allocator = match unsafe { vk_mem::Allocator::new(allocator_create_info) } {
			Ok(allocator) => allocator,
			Err(source) => {
				timeline.destroy(&handle);
				command_pool.destroy(&handle);
				// SAFETY: allocator creation failed and both existing device children were
				// destroyed. No callbacks were installed.
				unsafe {
					handle.destroy_device(None);
				}
				return Err(Error::backend_failure(
					"Vulkan",
					"memory-allocator creation",
					source,
				));
			}
		};
		let mut descriptors = match DescriptorHeap::new(&handle, physical.limits) {
			Ok(descriptors) => descriptors,
			Err(error) => {
				drop(allocator);
				timeline.destroy(&handle);
				command_pool.destroy(&handle);
				// SAFETY: every successfully created device child was destroyed above.
				unsafe {
					handle.destroy_device(None);
				}
				return Err(error);
			}
		};
		let mut pipelines = Vec::with_capacity(KernelId::ALL.len());
		for kernel in KernelId::ALL {
			match ComputePipeline::new(
				&handle,
				descriptors.layout(),
				kernel.artifact(),
				physical.limits,
			) {
				Ok(pipeline) => pipelines.push(pipeline),
				Err(error) => {
					for pipeline in &mut pipelines {
						pipeline.destroy(&handle);
					}
					descriptors.destroy(&handle);
					drop(allocator);
					timeline.destroy(&handle);
					command_pool.destroy(&handle);
					// SAFETY: every successfully created device child was destroyed above.
					unsafe {
						handle.destroy_device(None);
					}
					return Err(error);
				}
			}
		}

		Ok(Self {
			inner: Arc::new(DeviceInner {
				handle,
				allocator: ManuallyDrop::new(allocator),
				command_pool: Mutex::new(command_pool),
				video_decode_command_pool: Mutex::new(None),
				timeline,
				descriptors,
				storage_pool: Mutex::new(StoragePool::default()),
				pipelines,
				compute_queue,
				video_decode_queue,
				video_encode_queue,
				timestamp_period_ns: physical.limits.timestamp_period_ns,
				compute_timestamp_valid_bits: physical.limits.compute_timestamp_valid_bits,
				physical,
				_instance: instance.clone(),
			}),
		})
	}

	pub(in crate::runtime) fn physical(&self) -> &PhysicalDevice {
		&self.inner.physical
	}

	pub(super) fn instance(&self) -> &Instance {
		&self.inner._instance
	}

	#[cfg(test)]
	pub(in crate::runtime) fn create_video_decode_session(
		&self,
		profile: crate::video::VideoDecodeProfile,
		coded_extent: crate::video::VideoExtent,
		max_dpb_slots: u32,
		max_active_references: u32,
	) -> Result<super::video::DecodeSession> {
		super::video::create_decode_session(
			self,
			profile,
			coded_extent,
			max_dpb_slots,
			max_active_references,
		)
	}

	pub(super) fn allocator(&self) -> &vk_mem::Allocator {
		&self.inner.allocator
	}

	pub(super) fn bind_storage_buffer(
		&self,
		buffer: ash::vk::Buffer,
		size: ash::vk::DeviceSize,
	) -> Result<u32> {
		self.inner
			.descriptors
			.bind_storage_buffer(&self.inner.handle, buffer, size)
	}

	pub(super) fn release_storage_buffer(&self, index: u32) {
		self.inner.descriptors.release_storage_buffer(index);
	}

	pub(super) fn take_recycled_storage_buffer(&self, size: usize) -> Option<RecycledBuffer> {
		let mut pool = match self.inner.storage_pool.lock() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		let index = pool.free.iter().rposition(|buffer| buffer.size == size)?;
		let buffer = pool.free.swap_remove(index);
		pool.bytes = pool.bytes.saturating_sub(buffer.size);
		Some(buffer)
	}

	pub(super) fn recycle_storage_buffer(&self, buffer: RecycledBuffer) -> Option<RecycledBuffer> {
		let mut pool = match self.inner.storage_pool.lock() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		let Some(bytes) = pool.bytes.checked_add(buffer.size) else {
			return Some(buffer);
		};
		if pool.free.len() >= MAX_RECYCLED_STORAGE_BUFFERS || bytes > MAX_RECYCLED_STORAGE_BYTES {
			return Some(buffer);
		}
		pool.bytes = bytes;
		pool.free.push(buffer);
		None
	}

	pub(super) fn discard_one_recycled_storage_buffer(&self) -> bool {
		let buffer = {
			let mut pool = match self.inner.storage_pool.lock() {
				Ok(pool) => pool,
				Err(poisoned) => poisoned.into_inner(),
			};
			let Some(buffer) = pool.free.pop() else {
				return false;
			};
			pool.bytes = pool.bytes.saturating_sub(buffer.size);
			buffer
		};
		self.destroy_recycled_storage_buffer(buffer);
		true
	}

	fn destroy_recycled_storage_buffer(&self, mut buffer: RecycledBuffer) {
		self.release_storage_buffer(buffer.descriptor_index);
		// SAFETY: the pool exclusively owns this completed buffer/allocation pair,
		// and this device owns the allocator that created it.
		unsafe {
			self.allocator()
				.destroy_buffer(buffer.handle, &mut buffer.allocation);
		}
	}

	pub(in crate::runtime) fn same_as(&self, other: &Self) -> bool {
		Arc::ptr_eq(&self.inner, &other.inner)
	}

	pub(super) fn raw(&self) -> &ash::Device {
		&self.inner.handle
	}

	pub(in crate::runtime) fn record_empty(&self) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_empty(&self.inner.handle)
	}

	#[cfg(test)]
	pub(in crate::runtime) fn record_compute_commands(
		&self,
		record: impl FnOnce(ash::vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_custom(&self.inner.handle, record)
	}

	#[cfg(test)]
	pub(in crate::runtime) fn record_video_decode_empty(&self) -> Result<RecordedCommandBuffer> {
		self.record_video_decode(|_| Ok(()))
	}

	#[cfg(test)]
	pub(in crate::runtime) fn record_video_decode(
		&self,
		record: impl FnOnce(ash::vk::CommandBuffer) -> Result<()>,
	) -> Result<RecordedCommandBuffer> {
		let family = self
			.inner
			.physical
			.video
			.decode_queue_family
			.ok_or_else(|| Error::missing_capability("no Vulkan Video decode queue is enabled"))?;
		let mut slot = match self.inner.video_decode_command_pool.lock() {
			Ok(slot) => slot,
			Err(poisoned) => poisoned.into_inner(),
		};
		if slot.is_none() {
			*slot = Some(CommandPool::new(
				&self.inner.handle,
				family,
				CommandPoolKind::VideoDecode,
			)?);
		}
		slot.as_mut()
			.expect("video decode command pool was initialized")
			.record_custom(&self.inner.handle, record)
	}

	pub(in crate::runtime) fn record_compute_graph(
		&self,
		graph: &ExecutableGraph,
	) -> Result<RecordedCommandBuffer> {
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_compute_graph(
			&self.inner.handle,
			&self.inner.pipelines,
			self.inner.descriptors.set(),
			graph,
			None,
			None,
		)
	}

	pub(in crate::runtime) fn record_reusable_compute_graph(
		&self,
		graph: &ExecutableGraph,
	) -> Result<ReusableCommandBuffer> {
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		let command = command_pool.record_compute_graph(
			&self.inner.handle,
			&self.inner.pipelines,
			self.inner.descriptors.set(),
			graph,
			None,
			Some(self.clone()),
		)?;
		command.reusable().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"reusable command-buffer recording",
				std::io::Error::other("recording did not retain reusable ownership"),
			)
		})
	}

	pub(in crate::runtime) fn record_timed_compute_graph(
		&self,
		graph: &ExecutableGraph,
	) -> Result<RecordedCommandBuffer> {
		let timing = TimestampPair::new(
			self,
			self.inner.timestamp_period_ns,
			self.inner.compute_timestamp_valid_bits,
		)?;
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.record_compute_graph(
			&self.inner.handle,
			&self.inner.pipelines,
			self.inner.descriptors.set(),
			graph,
			Some(timing),
			None,
		)
	}

	pub(in crate::runtime) fn submit(
		&self,
		command: &RecordedCommandBuffer,
		epoch: u64,
	) -> Result<()> {
		let command_infos =
			[ash::vk::CommandBufferSubmitInfo::default().command_buffer(command.raw())];
		let signal_infos = [ash::vk::SemaphoreSubmitInfo::default()
			.semaphore(self.inner.timeline.raw())
			.value(epoch)
			.stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)];
		let wait_info = ash::vk::SemaphoreSubmitInfo::default()
			.semaphore(self.inner.timeline.raw())
			.value(epoch.saturating_sub(1))
			.stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS);
		let wait_infos = if epoch > 1 {
			std::slice::from_ref(&wait_info)
		} else {
			&[]
		};
		let submit_infos = [ash::vk::SubmitInfo2::default()
			.wait_semaphore_infos(wait_infos)
			.command_buffer_infos(&command_infos)
			.signal_semaphore_infos(&signal_infos)];
		let queue = match command.pool_kind() {
			CommandPoolKind::Compute => self.inner.compute_queue.handle,
			CommandPoolKind::VideoDecode => {
				self.inner
					.video_decode_queue
					.as_ref()
					.ok_or_else(|| {
						Error::missing_capability("no Vulkan Video decode queue is enabled")
					})?
					.handle
			}
		};

		// SAFETY: synchronization2 and timeline semaphores were enabled; the recorded
		// primary command buffer belongs to the selected queue family and is not pending.
		// The previous signal's first scope covers every earlier engine submission;
		// this submission waits on that exact timeline value at ALL_COMMANDS,
		// making writes to any Matrix input buffer available and visible before its
		// consumer dispatch. Buffer/image ownership transfers remain the recording
		// path's responsibility when queue families differ. Recorded commands retain
		// their resources until the signaled epoch is retired. The new signal uses the
		// next strictly increasing value.
		unsafe {
			self.inner
				.handle
				.queue_submit2(queue, &submit_infos, ash::vk::Fence::null())
		}
		.map_err(|source| Error::backend_failure("Vulkan", "queue submission", source))
	}

	pub(in crate::runtime) fn free(&self, command: RecordedCommandBuffer) {
		// Drop reusable ownership before taking the pool lock. Its final owner frees
		// the shared handle through this same mutex.
		let Some((handle, pool)) = command.into_owned_handle() else {
			return;
		};
		match pool {
			CommandPoolKind::Compute => {
				let mut command_pool = match self.inner.command_pool.lock() {
					Ok(command_pool) => command_pool,
					Err(poisoned) => poisoned.into_inner(),
				};
				command_pool.free_handle(&self.inner.handle, handle);
			}
			CommandPoolKind::VideoDecode => {
				let mut slot = match self.inner.video_decode_command_pool.lock() {
					Ok(slot) => slot,
					Err(poisoned) => poisoned.into_inner(),
				};
				if let Some(command_pool) = slot.as_mut() {
					command_pool.free_handle(&self.inner.handle, handle);
				}
			}
		}
	}

	pub(in crate::runtime) fn free_command_buffer_handle(&self, handle: ash::vk::CommandBuffer) {
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.free_handle(&self.inner.handle, handle);
	}

	pub(in crate::runtime) fn is_complete(&self, epoch: u64) -> Result<bool> {
		self.inner.timeline.is_complete(&self.inner.handle, epoch)
	}

	pub(in crate::runtime) fn wait(&self, epoch: u64) -> Result<()> {
		self.inner.timeline.wait(&self.inner.handle, epoch)
	}
}

impl Drop for DeviceInner {
	fn drop(&mut self) {
		let storage_pool = match self.storage_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		for mut buffer in storage_pool.free.drain(..) {
			// SAFETY: final device ownership proves no live buffer can reference these
			// pooled allocations, which were created by this allocator.
			unsafe {
				self.allocator
					.destroy_buffer(buffer.handle, &mut buffer.allocation);
			}
		}
		for pipeline in &mut self.pipelines {
			pipeline.destroy(&self.handle);
		}
		self.descriptors.destroy(&self.handle);
		// SAFETY: the final `Arc` proves no buffer owner remains. The allocator was
		// initialized exactly once and is dropped exactly once before its device.
		unsafe {
			ManuallyDrop::drop(&mut self.allocator);
		}
		self.timeline.destroy(&self.handle);
		let command_pool = match self.command_pool.get_mut() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.destroy(&self.handle);
		let video_decode_pool = match self.video_decode_command_pool.get_mut() {
			Ok(pool) => pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		if let Some(pool) = video_decode_pool.as_mut() {
			pool.destroy(&self.handle);
		}
		// SAFETY: the allocator, timeline, and command pools have been destroyed, and no
		// other logical-device children remain at this checkpoint.
		unsafe {
			self.handle.destroy_device(None);
		}
	}
}
