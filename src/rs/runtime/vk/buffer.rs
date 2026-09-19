use std::sync::{
	Arc, Mutex,
	atomic::{AtomicBool, Ordering},
};

use vk_mem::Alloc;

use crate::{Error, Result, core::memory};

use super::Device;

#[derive(Clone)]
pub(in crate::runtime) struct Buffer {
	inner: Arc<BufferInner>,
}

struct BufferInner {
	handle: ash::vk::Buffer,
	allocation: Mutex<Option<vk_mem::Allocation>>,
	recyclable: AtomicBool,
	size: usize,
	device_size: ash::vk::DeviceSize,
	descriptor_index: u32,
	device: Device,
}

pub(super) struct RecycledBuffer {
	pub(super) handle: ash::vk::Buffer,
	pub(super) allocation: vk_mem::Allocation,
	pub(super) size: usize,
	pub(super) device_size: ash::vk::DeviceSize,
	pub(super) descriptor_index: u32,
}

impl Buffer {
	pub(in crate::runtime) fn byte_len(&self) -> usize {
		self.inner.size
	}

	pub(in crate::runtime) fn host_visible_storage(device: &Device, size: usize) -> Result<Self> {
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
		let buffer_info = ash::vk::BufferCreateInfo::default()
			.size(device_size)
			.usage(
				ash::vk::BufferUsageFlags::STORAGE_BUFFER
					| ash::vk::BufferUsageFlags::TRANSFER_SRC
					| ash::vk::BufferUsageFlags::TRANSFER_DST,
			)
			.sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
		let allocation_info = vk_mem::AllocationCreateInfo {
			flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
			usage: vk_mem::MemoryUsage::AutoPreferDevice,
			required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
			preferred_flags: ash::vk::MemoryPropertyFlags::HOST_CACHED
				| ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
			..Default::default()
		};

		// SAFETY: both create structures are fully initialized, contain no extension
		// chains, and the allocator's instance/device owners remain live in `device`.
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.create_buffer(&buffer_info, &allocation_info)
		}
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
							device.allocator().destroy_buffer(handle, &mut allocation);
						}
						return Err(error);
					}
				}
			}
			Err(error) => {
				let mut allocation = allocation;
				unsafe {
					device.allocator().destroy_buffer(handle, &mut allocation);
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

	fn from_recycled(device: &Device, recycled: RecycledBuffer) -> Self {
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

	pub(in crate::runtime) fn host_visible_alias_arena(device: &Device, size: usize) -> Result<Self> {
		if size == 0 {
			return Err(Error::invalid_argument(
				"Vulkan alias arenas must be nonempty",
			));
		}
		let device_size = u64::try_from(size)
			.map_err(|_| Error::invalid_argument("alias buffer size exceeds Vulkan DeviceSize"))?;
		let buffer_info = storage_buffer_info(device_size);
		let allocation_info = vk_mem::AllocationCreateInfo {
			flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
			usage: vk_mem::MemoryUsage::AutoPreferDevice,
			required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE,
			preferred_flags: ash::vk::MemoryPropertyFlags::HOST_CACHED
				| ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
			..Default::default()
		};
		// SAFETY: both create structures are initialized and the retained Device
		// keeps the VMA allocator and Vulkan device alive for the allocation.
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.create_buffer(&buffer_info, &allocation_info)
		}
		.map_err(|source| Error::backend_failure("Vulkan", "alias backing allocation", source))?;
		let descriptor_index = match device.bind_storage_buffer(handle, device_size) {
			Ok(index) => index,
			Err(error) => {
				let mut allocation = allocation;
				// SAFETY: VMA returned this uniquely owned buffer/allocation pair and
				// descriptor publication failed before either escaped.
				unsafe {
					device.allocator().destroy_buffer(handle, &mut allocation);
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
		let mapped = unsafe { allocator.map_memory(allocation) }
			.map_err(|source| Error::backend_failure("Vulkan", "buffer mapping", source))?;

		// SAFETY: `validate_range` proved `[offset, offset + data.len())` is contained
		// in the mapped allocation. The byte slices cannot overlap because `data` is a
		// host borrow independent of OA's private allocation. GPU upload storage is a
		// one-way destination that the CPU will not consume before publication.
		unsafe {
			memory::copy_streaming_to_ptr(mapped.add(offset), data.as_ptr(), data.len());
		}

		let flush_result = allocator.flush_allocation(allocation, offset_u64, length_u64);
		// SAFETY: this exactly balances the successful map above, after the final host
		// write and flush attempt.
		unsafe {
			allocator.unmap_memory(allocation);
		}
		flush_result.map_err(|source| Error::backend_failure("Vulkan", "buffer flush", source))
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
		let mapped = unsafe { allocator.map_memory(allocation) }
			.map_err(|source| Error::backend_failure("Vulkan", "buffer mapping", source))?;
		let invalidate_result = allocator.invalidate_allocation(allocation, offset_u64, length_u64);

		if invalidate_result.is_ok() {
			// SAFETY: `validate_range` proved the source range is contained in the mapped
			// allocation. `output` is an independent mutable host slice of equal length.
			unsafe {
				memory::copy_to_ptr(output.as_mut_ptr(), mapped.add(offset), output.len());
			}
		}

		// SAFETY: this exactly balances the successful map above, after the final host
		// read or invalidate failure.
		unsafe {
			allocator.unmap_memory(allocation);
		}
		invalidate_result
			.map_err(|source| Error::backend_failure("Vulkan", "buffer invalidation", source))
	}

	pub(in crate::runtime) fn descriptor_index(&self) -> u32 {
		self.inner.descriptor_index
	}

	pub(in crate::runtime) fn same_as(&self, other: &Self) -> bool {
		Arc::ptr_eq(&self.inner, &other.inner)
	}

	pub(in crate::runtime) fn raw(&self) -> ash::vk::Buffer {
		self.inner.handle
	}

	pub(in crate::runtime) fn size(&self) -> ash::vk::DeviceSize {
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
				self
					.device
					.allocator()
					.destroy_buffer(self.handle, &mut allocation);
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
		// SAFETY: `Self` uniquely owns this buffer/allocation pair. Its `Device` keeps
		// the allocator alive throughout this call, and VMA created the pair together.
		unsafe {
			self
				.device
				.allocator()
				.destroy_buffer(recycled.handle, &mut recycled.allocation);
		}
	}
}

fn storage_buffer_info(size: ash::vk::DeviceSize) -> ash::vk::BufferCreateInfo<'static> {
	ash::vk::BufferCreateInfo::default()
		.size(size)
		.usage(
			ash::vk::BufferUsageFlags::STORAGE_BUFFER
				| ash::vk::BufferUsageFlags::TRANSFER_SRC
				| ash::vk::BufferUsageFlags::TRANSFER_DST,
		)
		.sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
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

#[cfg(test)]
mod tests {
	use super::validate_range;
	use crate::ErrorKind;

	#[test]
	fn validates_empty_and_boundary_ranges() {
		assert!(validate_range(16, 0, 0).is_ok());
		assert!(validate_range(16, 16, 0).is_ok());
		assert!(validate_range(16, 8, 8).is_ok());
	}

	#[test]
	fn rejects_overflow_and_out_of_bounds_ranges() {
		assert_eq!(
			validate_range(16, 15, 2).map_err(|error| error.kind()),
			Err(ErrorKind::InvalidArgument)
		);
		assert_eq!(
			validate_range(usize::MAX, usize::MAX, 1).map_err(|error| error.kind()),
			Err(ErrorKind::InvalidArgument)
		);
	}
}
