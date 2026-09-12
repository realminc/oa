use std::sync::Mutex;

use crate::{Error, Result};

use super::physical::DeviceLimits;

/// Engine-owned bindless descriptors shared by every compute pipeline.
pub(super) struct DescriptorHeap {
	pool: ash::vk::DescriptorPool,
	layout: ash::vk::DescriptorSetLayout,
	set: ash::vk::DescriptorSet,
	free_storage_indices: Mutex<Vec<u32>>,
}

impl DescriptorHeap {
	pub(super) fn new(device: &ash::Device, mut limits: DeviceLimits) -> Result<Self> {
		let minimum_capacity = limits.storage_buffer_descriptors.min(65_536);
		loop {
			match Self::try_new(device, limits) {
				Ok(heap) => return Ok(heap),
				Err(_) if limits.storage_buffer_descriptors > minimum_capacity => {
					limits.storage_buffer_descriptors =
						(limits.storage_buffer_descriptors / 2).max(minimum_capacity);
				}
				Err(error) => return Err(error),
			}
		}
	}

	fn try_new(device: &ash::Device, limits: DeviceLimits) -> Result<Self> {
		let binding = ash::vk::DescriptorSetLayoutBinding::default()
			.binding(0)
			.descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(limits.storage_buffer_descriptors)
			.stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
		let bindings = [binding];
		let binding_flags = [ash::vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
			| ash::vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
			| ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND];
		let mut binding_flags_info = ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
			.binding_flags(&binding_flags);
		let layout_info = ash::vk::DescriptorSetLayoutCreateInfo::default()
			.flags(ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
			.bindings(&bindings)
			.push_next(&mut binding_flags_info);
		// SAFETY: the single binding and its binding-flags entry have matching
		// counts, all required descriptor-indexing features were queried and enabled,
		// and no allocation callbacks are installed.
		let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }.map_err(
			|source| Error::backend_failure("Vulkan", "compute descriptor-layout creation", source),
		)?;

		let pool_size = ash::vk::DescriptorPoolSize::default()
			.ty(ash::vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(limits.storage_buffer_descriptors);
		let pool_sizes = [pool_size];
		let pool_info = ash::vk::DescriptorPoolCreateInfo::default()
			.flags(ash::vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
			.max_sets(1)
			.pool_sizes(&pool_sizes);
		// SAFETY: the requested storage-descriptor count is bounded by both queried
		// update-after-bind limits and the internal experimental capacity.
		let pool = match unsafe { device.create_descriptor_pool(&pool_info, None) } {
			Ok(pool) => pool,
			Err(source) => {
				destroy_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute descriptor-pool creation",
					source,
				));
			}
		};

		let layouts = [layout];
		let allocate_info = ash::vk::DescriptorSetAllocateInfo::default()
			.descriptor_pool(pool)
			.set_layouts(&layouts);
		// SAFETY: the pool has capacity for one set containing the complete layout.
		let sets = match unsafe { device.allocate_descriptor_sets(&allocate_info) } {
			Ok(sets) => sets,
			Err(source) => {
				destroy_pool(device, pool);
				destroy_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute descriptor-set allocation",
					source,
				));
			}
		};
		let Some(set) = sets.first().copied() else {
			destroy_pool(device, pool);
			destroy_layout(device, layout);
			return Err(Error::backend_failure(
				"Vulkan",
				"compute descriptor-set allocation",
				std::io::Error::other(
					"Vulkan returned no descriptor set after successful allocation",
				),
			));
		};

		Ok(Self {
			pool,
			layout,
			set,
			free_storage_indices: Mutex::new(
				(0..limits.storage_buffer_descriptors).rev().collect(),
			),
		})
	}

	pub(super) fn bind_storage_buffer(
		&self,
		device: &ash::Device,
		buffer: ash::vk::Buffer,
		size: ash::vk::DeviceSize,
	) -> Result<u32> {
		let mut free = self.free_storage_indices.lock().map_err(|_| {
			Error::backend_failure(
				"Vulkan",
				"storage-descriptor allocation",
				std::io::Error::other("storage descriptor allocator lock was poisoned"),
			)
		})?;
		let index = free.pop().ok_or_else(|| {
			Error::resource_exhausted("Vulkan storage descriptor capacity exhausted")
		})?;
		let buffer_info = ash::vk::DescriptorBufferInfo::default()
			.buffer(buffer)
			.offset(0)
			.range(size);
		let buffer_infos = [buffer_info];
		let write = ash::vk::WriteDescriptorSet::default()
			.dst_set(self.set)
			.dst_binding(0)
			.dst_array_element(index)
			.descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
			.buffer_info(&buffer_infos);
		// SAFETY: the descriptor set, buffer, and range are live. The allocator lock
		// serializes host updates, and update-after-bind plus update-unused-while-pending
		// permit a newly allocated, currently unused slot to be written while submitted
		// work may use other slots.
		unsafe {
			device.update_descriptor_sets(&[write], &[]);
		}
		Ok(index)
	}

	pub(super) fn release_storage_buffer(&self, index: u32) {
		let mut free = match self.free_storage_indices.lock() {
			Ok(free) => free,
			Err(poisoned) => poisoned.into_inner(),
		};
		free.push(index);
	}

	pub(super) fn destroy(&mut self, device: &ash::Device) {
		destroy_pool(device, self.pool);
		destroy_layout(device, self.layout);
	}

	pub(super) const fn layout(&self) -> ash::vk::DescriptorSetLayout {
		self.layout
	}

	pub(super) const fn set(&self) -> ash::vk::DescriptorSet {
		self.set
	}
}

fn destroy_pool(device: &ash::Device, pool: ash::vk::DescriptorPool) {
	// SAFETY: the pool belongs to `device`; destroying it releases its descriptor set.
	unsafe { device.destroy_descriptor_pool(pool, None) }
}

fn destroy_layout(device: &ash::Device, layout: ash::vk::DescriptorSetLayout) {
	// SAFETY: the layout belongs to `device` and no live set or pipeline layout uses it.
	unsafe { device.destroy_descriptor_set_layout(layout, None) }
}
