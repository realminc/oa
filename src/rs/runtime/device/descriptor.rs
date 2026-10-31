use std::sync::Arc;

use std::sync::{
	Mutex,
	atomic::{AtomicU32, Ordering},
};

use crate::{Error, Result};

use super::{features::ExecutionProfile, physical::DeviceLimits};
use crate::runtime::Device;

/// Engine-owned bindless descriptors shared by every compute and UI pipeline.
///
/// Layout (set=0):
/// - binding=0: STORAGE_BUFFER array (`heap[]`) — compute shader backing
/// - binding=1: STORAGE_IMAGE  array (`images[]`) — UI compose targets
pub(super) struct DescriptorHeap {
	profile: ExecutionProfile,
	pool: vk::DescriptorPool,
	layout: vk::DescriptorSetLayout,
	bounded_layouts: Vec<vk::DescriptorSetLayout>,
	set: vk::DescriptorSet,
	buffer_capacity: u32,
	image_capacity: u32,
	free_storage_indices: Mutex<Vec<u32>>,
	free_image_indices: Mutex<Vec<u32>>,
	virtual_next: AtomicU32,
}

impl DescriptorHeap {
	pub(super) fn capacities(&self) -> (u32, u32) {
		(self.buffer_capacity, self.image_capacity)
	}

	pub(super) fn new(
		device: &vk::Device,
		mut limits: DeviceLimits,
		profile: ExecutionProfile,
	) -> Result<Self> {
		if profile == ExecutionProfile::Compatibility {
			return Self::new_bounded(device, limits.storage_buffer_descriptors);
		}
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

	fn try_new(device: &vk::Device, limits: DeviceLimits) -> Result<Self> {
		let buffer_binding = vk::DescriptorSetLayoutBinding::default()
			.binding(0)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(limits.storage_buffer_descriptors)
			.stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT);
		let image_binding = vk::DescriptorSetLayoutBinding::default()
			.binding(1)
			.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
			.descriptor_count(limits.storage_image_descriptors)
			.stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT);
		let bindings = [buffer_binding, image_binding];
		let binding_flags = [
			vk::DescriptorBindingFlagBits::UPDATE_AFTER_BIND_BIT
				| vk::DescriptorBindingFlagBits::UPDATE_UNUSED_WHILE_PENDING_BIT
				| vk::DescriptorBindingFlagBits::PARTIALLY_BOUND_BIT,
			vk::DescriptorBindingFlagBits::UPDATE_AFTER_BIND_BIT
				| vk::DescriptorBindingFlagBits::UPDATE_UNUSED_WHILE_PENDING_BIT
				| vk::DescriptorBindingFlagBits::PARTIALLY_BOUND_BIT,
		];
		let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
			.binding_flags(&binding_flags)
			.map_err(Error::from)?;
		let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
			.flags(vk::DescriptorSetLayoutCreateFlagBits::UPDATE_AFTER_BIND_POOL_BIT)
			.bindings(&bindings)
			.map_err(Error::from)?
			.push_next(&mut binding_flags_info)
			.map_err(Error::from)?;
		// SAFETY: bindings and their binding-flags entries have matching counts, all
		// required descriptor-indexing features were queried and enabled, and no
		// allocation callbacks are installed.
		let layout =
			unsafe { device.create_descriptor_set_layout(&layout_info, None) }.map_err(|source| {
				Error::backend_failure("Vulkan", "compute descriptor-layout creation", source)
			})?;

		let pool_sizes = [
			vk::DescriptorPoolSize::default()
				.r#type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(limits.storage_buffer_descriptors),
			vk::DescriptorPoolSize::default()
				.r#type(vk::DescriptorType::STORAGE_IMAGE)
				.descriptor_count(limits.storage_image_descriptors),
		];
		let pool_info = vk::DescriptorPoolCreateInfo::default()
			.flags(vk::DescriptorPoolCreateFlagBits::UPDATE_AFTER_BIND_BIT)
			.max_sets(1)
			.pool_sizes(&pool_sizes)
			.map_err(Error::from)?;
		// SAFETY: the requested descriptor counts are bounded by queried update-after-bind limits.
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
		let allocate_info = vk::DescriptorSetAllocateInfo::default()
			.descriptor_pool(pool)
			.set_layouts(&layouts)
			.map_err(Error::from)?;
		let mut sets = [vk::DescriptorSet::default()];
		// SAFETY: the pool has capacity for one set containing the complete layout.
		match unsafe { device.allocate_descriptor_sets(&allocate_info, &mut sets) } {
			Ok(()) => {}
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
		let set = sets[0];

		Ok(Self {
			profile: ExecutionProfile::Strict,
			pool,
			layout,
			bounded_layouts: Vec::new(),
			set,
			buffer_capacity: limits.storage_buffer_descriptors,
			image_capacity: limits.storage_image_descriptors,
			free_storage_indices: Mutex::new((0..limits.storage_buffer_descriptors).rev().collect()),
			free_image_indices: Mutex::new((0..limits.storage_image_descriptors).rev().collect()),
			virtual_next: AtomicU32::new(0),
		})
	}

	fn new_bounded(device: &vk::Device, capacity: u32) -> Result<Self> {
		if capacity == 0 {
			return Err(Error::missing_capability(
				"bounded descriptors require a nonzero capacity",
			));
		}
		let mut bounded_layouts = Vec::new();
		for count in 1..=capacity {
			let binding = vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.descriptor_count(count)
				.stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT);
			let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
				.bindings(std::slice::from_ref(&binding))
				.map_err(Error::from)?;
			// SAFETY: each fixed array fits the queried per-stage/set budget;
			// the conventional binding needs no descriptor-indexing extension.
			// Uniform dynamic buffer-array indexing is enabled as a core feature.
			let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) };
			match layout {
				Ok(layout) => bounded_layouts.push(layout),
				Err(source) => {
					for layout in bounded_layouts {
						destroy_layout(device, layout);
					}
					return Err(Error::backend_failure(
						"Vulkan",
						"bounded descriptor-layout creation",
						source,
					));
				}
			}
		}
		let layout = bounded_layouts[0];
		// These are storage identities, not hardware descriptor indices. Recorded
		// commands map them to local indices and retain immutable descriptor sets.
		Ok(Self {
			profile: ExecutionProfile::Compatibility,
			pool: vk::DescriptorPool::default(),
			layout,
			bounded_layouts,
			set: vk::DescriptorSet::default(),
			buffer_capacity: capacity,
			image_capacity: 0,
			free_storage_indices: Mutex::new(Vec::new()),
			free_image_indices: Mutex::new(Vec::new()),
			virtual_next: AtomicU32::new(0),
		})
	}

	pub(super) fn bind_storage_buffer(
		&self,
		device: &vk::Device,
		buffer: vk::Buffer,
		size: vk::DeviceSize,
	) -> Result<u32> {
		if self.profile == ExecutionProfile::Compatibility {
			let _ = (device, buffer, size);
			if let Some(index) = self
				.free_storage_indices
				.lock()
				.map_err(|_| Error::internal("bounded descriptor identity allocator lock poisoned"))?
				.pop()
			{
				return Ok(index);
			}
			return self
				.virtual_next
				.try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
					next.checked_add(1)
				})
				.map_err(|_| Error::resource_exhausted("bounded storage identity space exhausted"));
		}
		let mut free = self.free_storage_indices.lock().map_err(|_| {
			Error::backend_failure(
				"Vulkan",
				"storage-descriptor allocation",
				std::io::Error::other("storage descriptor allocator lock was poisoned"),
			)
		})?;
		let index = free
			.pop()
			.ok_or_else(|| Error::resource_exhausted("Vulkan storage descriptor capacity exhausted"))?;
		let buffer_info = vk::DescriptorBufferInfo::default()
			.buffer(buffer)
			.offset(0)
			.range(size);
		let buffer_infos = [buffer_info];
		let write = vk::WriteDescriptorSet::default()
			.dst_set(self.set)
			.dst_binding(0)
			.dst_array_element(index)
			.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
			.buffer_info(&buffer_infos);
		// SAFETY: the descriptor set, buffer, and range are live. The allocator lock
		// serializes host updates, and update-after-bind plus update-unused-while-pending
		// permit a newly allocated, currently unused slot to be written while submitted
		// work may use other slots.
		unsafe {
			device
				.update_descriptor_sets(&[write.map_err(Error::from)?], &[])
				.map_err(Error::from)?;
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

	/// Allocate one STORAGE_IMAGE slot (binding=1) and write `view` into it.
	///
	/// The image must be in `GENERAL` layout when compute shaders read or write it.
	pub(super) fn bind_storage_image(&self, device: &vk::Device, view: vk::ImageView) -> Result<u32> {
		if self.profile == ExecutionProfile::Compatibility {
			// Bounded UI descriptors bind the retained view directly; there is no
			// global image array or descriptor slot to allocate.
			return Ok(0);
		}
		let mut free = self.free_image_indices.lock().map_err(|_| {
			Error::backend_failure(
				"Vulkan",
				"storage-image descriptor allocation",
				std::io::Error::other("storage image descriptor allocator lock was poisoned"),
			)
		})?;
		let index = free.pop().ok_or_else(|| {
			Error::resource_exhausted("Vulkan storage image descriptor capacity exhausted")
		})?;
		let image_info = vk::DescriptorImageInfo::default()
			.image_view(view)
			.image_layout(vk::ImageLayout::GENERAL);
		let image_infos = [image_info];
		let write = vk::WriteDescriptorSet::default()
			.dst_set(self.set)
			.dst_binding(1)
			.dst_array_element(index)
			.descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
			.image_info(&image_infos);
		unsafe {
			device
				.update_descriptor_sets(&[write.map_err(Error::from)?], &[])
				.map_err(Error::from)?;
		}
		Ok(index)
	}

	pub(super) fn release_storage_image(&self, index: u32) {
		if self.profile == ExecutionProfile::Compatibility {
			return;
		}
		let mut free = match self.free_image_indices.lock() {
			Ok(free) => free,
			Err(poisoned) => poisoned.into_inner(),
		};
		free.push(index);
	}

	pub(super) fn destroy(&mut self, device: &vk::Device) {
		if self.profile == ExecutionProfile::Strict {
			destroy_pool(device, self.pool);
			destroy_layout(device, self.layout);
		} else {
			for layout in self.bounded_layouts.drain(..) {
				destroy_layout(device, layout);
			}
		}
	}

	pub(super) fn bounded_layout(&self, count: u32) -> Option<vk::DescriptorSetLayout> {
		let index = usize::try_from(count.checked_sub(1)?).ok()?;
		self.bounded_layouts.get(index).copied()
	}

	pub(super) const fn layout(&self) -> vk::DescriptorSetLayout {
		self.layout
	}

	pub(super) const fn set(&self) -> vk::DescriptorSet {
		self.set
	}
}

fn destroy_pool(device: &vk::Device, pool: vk::DescriptorPool) {
	// SAFETY: the pool belongs to `device`; destroying it releases its descriptor set.
	// A missing destroy entry point cannot be recovered here; destruction is final.
	let _ = unsafe { device.destroy_descriptor_pool(pool, None) };
}

fn destroy_layout(device: &vk::Device, layout: vk::DescriptorSetLayout) {
	// SAFETY: the layout belongs to `device` and no live set or pipeline layout uses it.
	// A missing destroy entry point cannot be recovered here; destruction is final.
	let _ = unsafe { device.destroy_descriptor_set_layout(layout, None) };
}

/// Immutable descriptor sets owned by one recorded compute graph. The pool is
/// destroyed only when its exact recorded command has retired (or its last
/// reusable owner is dropped after submissions complete).
pub(super) struct BoundedSets {
	device: Arc<Device>,
	pool: vk::DescriptorPool,
	sets: Vec<vk::DescriptorSet>,
}

impl BoundedSets {
	pub(super) fn new(
		context: &Arc<Device>,
		layouts: &[vk::DescriptorSetLayout],
		buffers: &[Vec<vk::DescriptorBufferInfo>],
	) -> Result<Self> {
		let device = context.raw();
		if layouts.is_empty() || layouts.len() != buffers.len() {
			return Err(Error::internal(
				"bounded descriptor layouts and dispatches must be nonempty and match",
			));
		}
		for (&layout, infos) in layouts.iter().zip(buffers) {
			let required = u32::try_from(infos.len())
				.map_err(|_| Error::resource_exhausted("bounded dispatch descriptor count overflow"))?;
			if context.bounded_descriptor_layout(required) != Some(layout) {
				return Err(Error::internal(
					"bounded descriptor layout does not match dispatch",
				));
			}
		}
		let count = u32::try_from(buffers.len())
			.map_err(|_| Error::resource_exhausted("too many bounded compute dispatches"))?;
		let descriptor_count = buffers.iter().try_fold(0_u32, |total, infos| {
			let len = u32::try_from(infos.len())
				.map_err(|_| Error::resource_exhausted("bounded dispatch descriptor count overflow"))?;
			total
				.checked_add(len)
				.ok_or_else(|| Error::resource_exhausted("bounded descriptor count overflow"))
		})?;
		let sizes = [vk::DescriptorPoolSize::default()
			.r#type(vk::DescriptorType::STORAGE_BUFFER)
			.descriptor_count(descriptor_count)];
		let info = vk::DescriptorPoolCreateInfo::default()
			.max_sets(count)
			.pool_sizes(&sizes)
			.map_err(Error::from)?;
		// SAFETY: the nonzero pool sizes exactly sum the validated live layouts;
		// no optional pool flags or allocation callbacks are used.
		let pool = unsafe { device.create_descriptor_pool(&info, None) }.map_err(|source| {
			Error::backend_failure("Vulkan", "bounded descriptor-pool creation", source)
		})?;
		let allocate = vk::DescriptorSetAllocateInfo::default()
			.descriptor_pool(pool)
			.set_layouts(layouts)
			.map_err(Error::from)?;
		// SAFETY: every layout belongs to this device and the new pool has the
		// exact set/descriptor capacity. All arrays live throughout the call.
		let mut sets = vec![vk::DescriptorSet::default(); layouts.len()];
		match unsafe { device.allocate_descriptor_sets(&allocate, &mut sets) } {
			Ok(()) => {}
			Err(source) => {
				destroy_pool(device, pool);
				return Err(Error::backend_failure(
					"Vulkan",
					"bounded descriptor-set allocation",
					source,
				));
			}
		};
		for (&set, infos) in sets.iter().zip(buffers) {
			let write = vk::WriteDescriptorSet::default()
				.dst_set(set)
				.dst_binding(0)
				.descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
				.buffer_info(infos);
			// SAFETY: every slot has a retained live buffer/range; this fresh set
			// has not been bound or submitted and its capacity matches `infos`.
			unsafe {
				device
					.update_descriptor_sets(&[write.map_err(Error::from)?], &[])
					.map_err(Error::from)?
			};
		}
		Ok(Self {
			device: context.clone(),
			pool,
			sets,
		})
	}

	pub(super) fn set(&self, index: usize) -> vk::DescriptorSet {
		self.sets[index]
	}
}

impl Drop for BoundedSets {
	fn drop(&mut self) {
		destroy_pool(self.device.raw(), self.pool);
	}
}
