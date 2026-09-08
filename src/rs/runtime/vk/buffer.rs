use std::sync::{Arc, Mutex};

use vk_mem::Alloc;

use crate::{Error, Result};

use super::Device;

#[derive(Clone)]
pub(in crate::runtime) struct Buffer {
	inner: Arc<BufferInner>,
}

struct BufferInner {
	handle: ash::vk::Buffer,
	allocation: Mutex<vk_mem::Allocation>,
	size: usize,
	device_size: ash::vk::DeviceSize,
	descriptor_index: u32,
	device: Device,
}

impl Buffer {
	pub(in crate::runtime) fn host_visible_storage(device: &Device, size: usize) -> Result<Self> {
		if size == 0 {
			return Err(Error::invalid_argument(
				"Vulkan buffers must contain at least one byte",
			));
		}

		let device_size = u64::try_from(size)
			.map_err(|_| Error::invalid_argument("buffer size exceeds Vulkan DeviceSize"))?;
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
				allocation: Mutex::new(allocation),
				size,
				device_size,
				descriptor_index,
				device: device.clone(),
			}),
		})
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
		let mapped = unsafe { allocator.map_memory(&mut allocation) }
			.map_err(|source| Error::backend_failure("Vulkan", "buffer mapping", source))?;

		// SAFETY: `validate_range` proved `[offset, offset + data.len())` is contained
		// in the mapped allocation. The byte slices cannot overlap because `data` is a
		// host borrow independent of OA's private allocation.
		unsafe {
			mapped
				.add(offset)
				.copy_from_nonoverlapping(data.as_ptr(), data.len());
		}

		let flush_result = allocator.flush_allocation(&allocation, offset_u64, length_u64);
		// SAFETY: this exactly balances the successful map above, after the final host
		// write and flush attempt.
		unsafe {
			allocator.unmap_memory(&mut allocation);
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
		let mapped = unsafe { allocator.map_memory(&mut allocation) }
			.map_err(|source| Error::backend_failure("Vulkan", "buffer mapping", source))?;
		let invalidate_result =
			allocator.invalidate_allocation(&allocation, offset_u64, length_u64);

		if invalidate_result.is_ok() {
			// SAFETY: `validate_range` proved the source range is contained in the mapped
			// allocation. `output` is an independent mutable host slice of equal length.
			unsafe {
				output
					.as_mut_ptr()
					.copy_from_nonoverlapping(mapped.add(offset), output.len());
			}
		}

		// SAFETY: this exactly balances the successful map above, after the final host
		// read or invalidate failure.
		unsafe {
			allocator.unmap_memory(&mut allocation);
		}
		invalidate_result
			.map_err(|source| Error::backend_failure("Vulkan", "buffer invalidation", source))
	}

	pub(in crate::runtime) fn descriptor_index(&self) -> u32 {
		self.inner.descriptor_index
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
		self.device.release_storage_buffer(self.descriptor_index);
		let allocation = match self.allocation.get_mut() {
			Ok(allocation) => allocation,
			Err(poisoned) => poisoned.into_inner(),
		};
		// SAFETY: `Self` uniquely owns this buffer/allocation pair. Its `Device` keeps
		// the allocator alive throughout this call, and VMA created the pair together.
		unsafe {
			self.device
				.allocator()
				.destroy_buffer(self.handle, allocation);
		}
	}
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
