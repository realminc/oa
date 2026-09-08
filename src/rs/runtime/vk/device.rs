use std::{
	mem::ManuallyDrop,
	sync::{Arc, Mutex},
};

use crate::{
	Error, Result,
	runtime::{executable_graph::ExecutableGraph, shader::KernelId},
};

use super::{
	Instance, PhysicalDevice,
	command::{CommandPool, RecordedCommandBuffer, ReusableCommandBuffer},
	descriptor::DescriptorHeap,
	pipeline::ComputePipeline,
	queue::Queue,
	timeline::Timeline,
	timestamp::TimestampPair,
};

#[derive(Clone)]
pub(in crate::runtime) struct Device {
	inner: Arc<DeviceInner>,
}

struct DeviceInner {
	handle: ash::Device,
	allocator: ManuallyDrop<vk_mem::Allocator>,
	command_pool: Mutex<CommandPool>,
	timeline: Timeline,
	descriptors: DescriptorHeap,
	pipelines: Vec<ComputePipeline>,
	_physical: PhysicalDevice,
	compute_queue: Queue,
	timestamp_period_ns: f64,
	compute_timestamp_valid_bits: u32,
	_instance: Instance,
}

impl Device {
	pub(in crate::runtime) fn new(instance: &Instance, physical: PhysicalDevice) -> Result<Self> {
		let priorities = [1.0_f32];
		let queue_create_infos = [ash::vk::DeviceQueueCreateInfo::default()
			.queue_family_index(physical.compute_queue_family)
			.queue_priorities(&priorities)];
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
		let mut command_pool = match CommandPool::new(&handle, physical.compute_queue_family) {
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
				timeline,
				descriptors,
				pipelines,
				_physical: physical,
				compute_queue,
				timestamp_period_ns: physical.limits.timestamp_period_ns,
				compute_timestamp_valid_bits: physical.limits.compute_timestamp_valid_bits,
				_instance: instance.clone(),
			}),
		})
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

		// SAFETY: synchronization2 and timeline semaphores were enabled; the recorded
		// primary command buffer belongs to this queue family and is not pending. The
		// previous signal's first scope covers every earlier operation on this compute
		// queue; this submission waits on that exact timeline value at ALL_COMMANDS,
		// making writes to any Matrix input buffer available and visible before its
		// consumer dispatch. Producer and consumer use the same queue family, buffers
		// have no image layout, and no ownership transfer is required. Recorded commands
		// retain those buffers until the signaled epoch is retired. The new signal uses
		// the next strictly increasing value.
		unsafe {
			self.inner.handle.queue_submit2(
				self.inner.compute_queue.handle,
				&submit_infos,
				ash::vk::Fence::null(),
			)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "queue submission", source))
	}

	pub(in crate::runtime) fn free(&self, command: RecordedCommandBuffer) {
		// Drop reusable ownership before taking the pool lock. Its final owner frees
		// the shared handle through this same mutex.
		let Some(handle) = command.into_owned_handle() else {
			return;
		};
		let mut command_pool = match self.inner.command_pool.lock() {
			Ok(command_pool) => command_pool,
			Err(poisoned) => poisoned.into_inner(),
		};
		command_pool.free_handle(&self.inner.handle, handle);
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
		// SAFETY: the allocator, timeline, and command pool have been destroyed, and no
		// other logical-device children remain at this checkpoint.
		unsafe {
			self.handle.destroy_device(None);
		}
	}
}
