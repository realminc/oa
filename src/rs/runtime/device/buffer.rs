use std::sync::{
	Arc, Mutex,
	atomic::{AtomicBool, Ordering},
};

use crate::{Error, Result, core::memory};

use crate::runtime::Device;

#[derive(Clone)]
pub(in crate::runtime) struct Buffer {
	inner: Arc<BufferInner>,
}

struct BufferInner {
	handle: vk::Buffer,
	allocation: Mutex<Option<vk::mem::Allocation>>,
	recyclable: AtomicBool,
	size: usize,
	device_size: vk::DeviceSize,
	descriptor_index: u32,
	device: Arc<Device>,
}

pub(super) struct RecycledBuffer {
	pub(super) handle: vk::Buffer,
	pub(super) allocation: vk::mem::Allocation,
	pub(super) size: usize,
	pub(super) device_size: vk::DeviceSize,
	pub(super) descriptor_index: u32,
}

impl Buffer {
	pub(in crate::runtime) fn byte_len(&self) -> usize {
		self.inner.size
	}

	pub(in crate::runtime) fn host_visible_storage(
		device: &Arc<Device>,
		size: usize,
	) -> Result<Self> {
		if size == 0 {
			return Err(Error::invalid_argument(
				"Vulkan buffers must contain at least one byte",
			));
		}

		let device_size = u64::try_from(size)
			.map_err(|_| Error::invalid_argument("buffer size exceeds Vulkan DeviceSize"))?;
		if let Some(recycled) = device.take_recycled_storage_buffer(size) {
			return Ok(Self::from_recycled(device, recycled));
		}
		let buffer_info = vk::BufferCreateInfo::new()
			.set_size(device_size)
			.set_usage(
				vk::BufferUsageFlagBits::STORAGE_BUFFER_BIT
					| vk::BufferUsageFlagBits::TRANSFER_SRC_BIT
					| vk::BufferUsageFlagBits::TRANSFER_DST_BIT
					| vk::BufferUsageFlagBits::INDIRECT_BUFFER_BIT,
			)
			.set_sharing_mode(vk::SharingMode::EXCLUSIVE);
		let allocation_info = vk::mem::AllocationCreateInfo {
			host_access: vk::mem::HostAccess::Random,
			usage: vk::mem::MemoryUsage::PreferDevice,
			required_flags: vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT,
			preferred_flags: vk::MemoryPropertyFlagBits::HOST_CACHED_BIT
				| vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT,
			..Default::default()
		};

		// SAFETY: both create structures are fully initialized, contain no extension
		// chains, and the allocator's instance/device owners remain live in `device`.
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.with_allocator(|a| a.create_buffer_with_options(&buffer_info, &allocation_info))
		}
		.map_err(|source| Error::backend_failure("Vulkan", "storage-buffer allocation", source))?
		.map_err(|source| Error::backend_failure("Vulkan", "storage-buffer allocation", source))?;

		let descriptor_index = match device.bind_storage_buffer(handle, device_size) {
			Ok(index) => index,
			Err(error)
				if error.kind() == crate::ErrorKind::ResourceExhausted
					&& device.discard_one_recycled_storage_buffer() =>
			{
				match device.bind_storage_buffer(handle, device_size) {
					Ok(index) => index,
					Err(error) => {
						let mut allocation = allocation;
						unsafe {
							device
								.allocator()
								.with_allocator(|a| a.destroy_buffer(handle, &mut allocation))??;
						}
						return Err(error);
					}
				}
			}
			Err(error) => {
				let mut allocation = allocation;
				unsafe {
					device
						.allocator()
						.with_allocator(|a| a.destroy_buffer(handle, &mut allocation))??;
				}
				return Err(error);
			}
		};

		Ok(Self {
			inner: Arc::new(BufferInner {
				handle,
				allocation: Mutex::new(Some(allocation)),
				recyclable: AtomicBool::new(true),
				size,
				device_size,
				descriptor_index,
				device: device.clone(),
			}),
		})
	}

	fn from_recycled(device: &Arc<Device>, recycled: RecycledBuffer) -> Self {
		Self {
			inner: Arc::new(BufferInner {
				handle: recycled.handle,
				allocation: Mutex::new(Some(recycled.allocation)),
				recyclable: AtomicBool::new(true),
				size: recycled.size,
				device_size: recycled.device_size,
				descriptor_index: recycled.descriptor_index,
				device: device.clone(),
			}),
		}
	}

	pub(in crate::runtime) fn host_visible_alias_arena(
		device: &Arc<Device>,
		size: usize,
	) -> Result<Self> {
		if size == 0 {
			return Err(Error::invalid_argument(
				"Vulkan alias arenas must be nonempty",
			));
		}
		let device_size = u64::try_from(size)
			.map_err(|_| Error::invalid_argument("alias buffer size exceeds Vulkan DeviceSize"))?;
		let buffer_info = storage_buffer_info(device_size);
		let allocation_info = vk::mem::AllocationCreateInfo {
			host_access: vk::mem::HostAccess::Random,
			usage: vk::mem::MemoryUsage::PreferDevice,
			required_flags: vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT,
			preferred_flags: vk::MemoryPropertyFlagBits::HOST_CACHED_BIT
				| vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT,
			..Default::default()
		};
		// SAFETY: both create structures are initialized and the retained Device
		// keeps the VMA allocator and Vulkan device alive for the allocation.
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.with_allocator(|a| a.create_buffer_with_options(&buffer_info, &allocation_info))
		}
		.map_err(|source| Error::backend_failure("Vulkan", "alias backing allocation", source))?
		.map_err(|source| Error::backend_failure("Vulkan", "alias backing allocation", source))?;
		let descriptor_index = match device.bind_storage_buffer(handle, device_size) {
			Ok(index) => index,
			Err(error) => {
				let mut allocation = allocation;
				// SAFETY: VMA returned this uniquely owned buffer/allocation pair and
				// descriptor publication failed before either escaped.
				unsafe {
					device
						.allocator()
						.with_allocator(|a| a.destroy_buffer(handle, &mut allocation))??;
				}
				return Err(error);
			}
		};
		Ok(Self {
			inner: Arc::new(BufferInner {
				handle,
				allocation: Mutex::new(Some(allocation)),
				recyclable: AtomicBool::new(false),
				size,
				device_size,
				descriptor_index,
				device: device.clone(),
			}),
		})
	}

	pub(in crate::runtime) fn disable_recycling(&self) {
		self.inner.recyclable.store(false, Ordering::Release);
	}

	pub(in crate::runtime) fn write(&self, offset: usize, data: &[u8]) -> Result<()> {
		validate_range(self.inner.size, offset, data.len())?;
		if data.is_empty() {
			return Ok(());
		}
		let offset_u64 = u64::try_from(offset)
			.map_err(|_| Error::invalid_argument("buffer offset exceeds Vulkan DeviceSize"))?;
		let length_u64 = u64::try_from(data.len())
			.map_err(|_| Error::invalid_argument("buffer length exceeds Vulkan DeviceSize"))?;

		let allocator = self.inner.device.allocator();
		let mut allocation = self.inner.allocation.lock().map_err(|_| {
			Error::backend_failure(
				"Vulkan",
				"buffer allocation lock",
				std::io::Error::other("buffer allocation lock was poisoned"),
			)
		})?;
		// SAFETY: the allocation lock excludes every other OA mapping operation and
		// VMA selected HOST_VISIBLE memory.
		let allocation = allocation.as_mut().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"buffer mapping",
				std::io::Error::other("buffer allocation was already recycled"),
			)
		})?;
		let mapped = unsafe { allocator.with_allocator(|a| a.map(allocation)) }
			.map_err(|source| Error::backend_failure("Vulkan", "buffer mapping", source))?;

		// SAFETY: `validate_range` proved `[offset, offset + data.len())` is contained
		// in the mapped allocation. The byte slices cannot overlap because `data` is a
		// host borrow independent of OA's private allocation. GPU upload storage is a
		// one-way destination that the CPU will not consume before publication.
		unsafe {
			memory::copy_streaming_to_ptr(
				(mapped.map_err(Error::from)? as *mut u8).add(offset),
				data.as_ptr(),
				data.len(),
			);
		}

		// SAFETY: same mapping contract as `map` above; the range was validated
		// against the allocation size and the allocation lock is held.
		let flush_result =
			unsafe { allocator.with_allocator(|a| a.flush(allocation, offset_u64, length_u64)) };
		// SAFETY: this exactly balances the successful map above, after the final host
		// write and flush attempt.
		unsafe {
			allocator.with_allocator(|a| a.unmap(allocation))??;
		}
		flush_result
			.map_err(|source| Error::backend_failure("Vulkan", "buffer flush", source))?
			.map_err(|source| Error::backend_failure("Vulkan", "buffer flush", source))
	}

	pub(in crate::runtime) fn read(&self, offset: usize, output: &mut [u8]) -> Result<()> {
		validate_range(self.inner.size, offset, output.len())?;
		if output.is_empty() {
			return Ok(());
		}
		let offset_u64 = u64::try_from(offset)
			.map_err(|_| Error::invalid_argument("buffer offset exceeds Vulkan DeviceSize"))?;
		let length_u64 = u64::try_from(output.len())
			.map_err(|_| Error::invalid_argument("buffer length exceeds Vulkan DeviceSize"))?;

		let allocator = self.inner.device.allocator();
		let mut allocation = self.inner.allocation.lock().map_err(|_| {
			Error::backend_failure(
				"Vulkan",
				"buffer allocation lock",
				std::io::Error::other("buffer allocation lock was poisoned"),
			)
		})?;
		// SAFETY: the allocation lock excludes every other OA mapping operation and
		// VMA selected HOST_VISIBLE memory.
		let allocation = allocation.as_mut().ok_or_else(|| {
			Error::backend_failure(
				"Vulkan",
				"buffer mapping",
				std::io::Error::other("buffer allocation was already recycled"),
			)
		})?;
		let mapped = unsafe { allocator.with_allocator(|a| a.map(allocation)) }
			.map_err(|source| Error::backend_failure("Vulkan", "buffer mapping", source))?;
		// SAFETY: same mapping contract as `map` above; the range was validated
		// against the allocation size and the allocation lock is held.
		let invalidate_result =
			unsafe { allocator.with_allocator(|a| a.invalidate(allocation, offset_u64, length_u64)) };

		if matches!(invalidate_result, Ok(Ok(()))) {
			// SAFETY: `validate_range` proved the source range is contained in the mapped
			// allocation. `output` is an independent mutable host slice of equal length.
			unsafe {
				memory::copy_to_ptr(
					output.as_mut_ptr(),
					(mapped.map_err(Error::from)? as *const u8).add(offset),
					output.len(),
				);
			}
		}

		// SAFETY: this exactly balances the successful map above, after the final host
		// read or invalidate failure.
		unsafe {
			allocator.with_allocator(|a| a.unmap(allocation))??;
		}
		invalidate_result
			.map_err(|source| Error::backend_failure("Vulkan", "buffer invalidation", source))?
			.map_err(|source| Error::backend_failure("Vulkan", "buffer invalidation", source))
	}

	pub(in crate::runtime) fn descriptor_index(&self) -> u32 {
		self.inner.descriptor_index
	}

	pub(in crate::runtime) fn belongs_to(&self, device: &Device) -> bool {
		self.inner.device.same_as(device)
	}

	pub(in crate::runtime) fn device(&self) -> &Arc<Device> {
		&self.inner.device
	}

	pub(in crate::runtime) fn same_as(&self, other: &Self) -> bool {
		Arc::ptr_eq(&self.inner, &other.inner)
	}

	pub(in crate::runtime) fn raw(&self) -> vk::Buffer {
		self.inner.handle
	}

	pub(in crate::runtime) fn size(&self) -> vk::DeviceSize {
		self.inner.device_size
	}
}

impl Drop for BufferInner {
	fn drop(&mut self) {
		let allocation = match self.allocation.get_mut() {
			Ok(allocation) => allocation,
			Err(poisoned) => poisoned.into_inner(),
		};
		let Some(allocation) = allocation.take() else {
			return;
		};
		if !self.recyclable.load(Ordering::Acquire) {
			self.device.release_storage_buffer(self.descriptor_index);
			let mut allocation = allocation;
			// SAFETY: final Buffer ownership uniquely owns the VMA-created pair.
			unsafe {
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.destroy_buffer(self.handle, &mut allocation));
			}
			return;
		}
		let recycled = RecycledBuffer {
			handle: self.handle,
			allocation,
			size: self.size,
			device_size: self.device_size,
			descriptor_index: self.descriptor_index,
		};
		let Some(mut recycled) = self.device.recycle_storage_buffer(recycled) else {
			return;
		};
		self
			.device
			.release_storage_buffer(recycled.descriptor_index);
		// SAFETY: `Self` uniquely owns this buffer/allocation pair. Its retained `Device` keeps
		// the allocator alive throughout this call, and VMA created the pair together.
		unsafe {
			let _ = self
				.device
				.allocator()
				.destroy_buffer(recycled.handle, &mut recycled.allocation);
		}
	}
}

fn storage_buffer_info(size: vk::DeviceSize) -> vk::BufferCreateInfo<'static> {
	vk::BufferCreateInfo::new()
		.set_size(size)
		.set_usage(
			vk::BufferUsageFlagBits::STORAGE_BUFFER_BIT
				| vk::BufferUsageFlagBits::TRANSFER_SRC_BIT
				| vk::BufferUsageFlagBits::TRANSFER_DST_BIT
				| vk::BufferUsageFlagBits::INDIRECT_BUFFER_BIT,
		)
		.set_sharing_mode(vk::SharingMode::EXCLUSIVE)
}

fn validate_range(size: usize, offset: usize, length: usize) -> Result<()> {
	let end = offset
		.checked_add(length)
		.ok_or_else(|| Error::invalid_argument("buffer range overflows usize"))?;
	if end > size {
		return Err(Error::invalid_argument(format!(
			"buffer range {offset}..{end} exceeds allocation size {size}"
		)));
	}
	Ok(())
}

/// Host-visible, persistently mapped buffer for mesh vertex or index data.
///
/// Mesh buffers are bound with `vkCmdBindVertexBuffers` /
/// `vkCmdBindIndexBuffer` and never enter the bindless descriptor heap —
/// they are not storage buffers and are not accessed by compute shaders.
///
/// Donor: `oa::MeshBuffer` / `oa::FnMesh`.
pub(crate) struct MeshBuffer {
	device: Arc<Device>,
	handle: vk::Buffer,
	allocation: Option<vk::mem::Allocation>,
	/// Persistently mapped host-write pointer into the allocation.
	mapped_ptr: *mut u8,
	pub(crate) capacity: usize,
}

impl MeshBuffer {
	/// Allocate a host-visible, persistently mapped vertex or index buffer.
	pub(in crate::runtime) fn new(
		device: &Arc<Device>,
		size: usize,
		usage: vk::BufferUsageFlags,
	) -> Result<Self> {
		if size == 0 {
			return Err(Error::invalid_argument("MeshBuffer size must be non-zero"));
		}
		let device_size = u64::try_from(size)
			.map_err(|_| Error::invalid_argument("geometry buffer size exceeds u64"))?;
		let buf_info = vk::BufferCreateInfo::new()
			.set_size(device_size)
			.set_usage(usage)
			.set_sharing_mode(vk::SharingMode::EXCLUSIVE);
		let alloc_info = vk::mem::AllocationCreateInfo {
			host_access: vk::mem::HostAccess::SequentialWrite,
			persistent_map: true,
			usage: vk::mem::MemoryUsage::PreferDevice,
			required_flags: vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT,
			..Default::default()
		};
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.with_allocator(|a| a.create_buffer_with_options(&buf_info, &alloc_info))
		}
		.map_err(|s| Error::backend_failure("Vulkan", "geometry buffer allocation", s))?
		.map_err(|s| Error::backend_failure("Vulkan", "geometry buffer allocation", s))?;
		// SAFETY: `allocation` was just created by this same allocator and is
		// uniquely owned here.
		let info = unsafe {
			device
				.allocator()
				.with_allocator(|a| a.allocation_info(&allocation))
		}
		.map_err(|s| Error::backend_failure("Vulkan", "geometry allocation info", s))?
		.map_err(|s| Error::backend_failure("Vulkan", "geometry allocation info", s))?;
		let mapped_ptr = info
			.mapped_pointer
			.map_or(std::ptr::null_mut(), |ptr| ptr.cast::<u8>());
		if mapped_ptr.is_null() {
			let mut alloc = allocation;
			unsafe {
				device
					.allocator()
					.with_allocator(|a| a.destroy_buffer(handle, &mut alloc))
			}??;
			return Err(Error::backend_failure(
				"Vulkan",
				"geometry buffer persistent map",
				std::io::Error::other("VMA returned null mapped pointer"),
			));
		}
		Ok(Self {
			device: device.clone(),
			handle,
			allocation: Some(allocation),
			mapped_ptr,
			capacity: size,
		})
	}

	/// Write `data` into the mapped buffer at offset 0 and flush.
	pub(crate) fn write_and_flush(&self, data: &[u8]) -> Result<()> {
		if data.len() > self.capacity {
			return Err(Error::invalid_argument("mesh data exceeds buffer capacity"));
		}
		// SAFETY: mapped_ptr is valid for `capacity` bytes for the lifetime of
		// this buffer, and data fits within that range.
		unsafe {
			std::ptr::copy_nonoverlapping(data.as_ptr(), self.mapped_ptr, data.len());
		}
		let alloc = self.allocation.as_ref().expect("mesh allocation live");
		let size = u64::try_from(data.len()).unwrap_or(u64::MAX);
		unsafe {
			self
				.device
				.allocator()
				.with_allocator(|a| a.flush(alloc, 0, size))
		}
		.map_err(|s| Error::backend_failure("Vulkan", "mesh buffer flush", s))?
		.map_err(|s| Error::backend_failure("Vulkan", "mesh buffer flush", s))
	}

	pub(crate) const fn raw(&self) -> vk::Buffer {
		self.handle
	}
}

impl Drop for MeshBuffer {
	fn drop(&mut self) {
		if let Some(mut a) = self.allocation.take() {
			let _ = // SAFETY: buffer was created from this device's allocator and is no
			// longer in use (the renderer waited for the producer event before drop).
			unsafe { self.device.allocator().with_allocator(|allocator| allocator.destroy_buffer(self.handle, &mut a)) };
		}
	}
}

// SAFETY: MeshBuffer is not Send/Sync by default because of the raw pointer,
// but EngineHandle and all of our render types are already !Send (they use Rc).
// Mark it explicitly so the compiler doesn't warn.
