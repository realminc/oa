//! Private device secrets and separate locked upload staging.
//! No public mapping, export, Matrix conversion or Clone.
//!
//! New ownership implementation: the C++ oaPqc.md donor has no secret-device
//! allocator to port. Implements oaGpuSecretSecurity.md §6, not algorithm admission.

use std::{
	mem,
	sync::{
		Arc,
		atomic::{AtomicU8, Ordering},
	},
};

use vk_mem::Alloc;

use crate::{Error, Result, runtime::Event};

use crate::runtime::{Device, RecordedCommandBuffer};

const LIVE: u8 = 0;
const PENDING: u8 = 1;
const CONFIRMED: u8 = 2;
const UNCONFIRMED: u8 = 3;
const GPU_COMPLETE: u8 = 4;

pub(in crate::runtime) struct SecretBuffer {
	device: Arc<Device>,
	handle: ash::vk::Buffer,
	allocation: Option<vk_mem::Allocation>,
	size: u64,
	descriptor_index: Option<u32>,
	state: Arc<AtomicU8>,
	host_mapping: Option<HostMapping>,
	memory: SecretMemory,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SecretMemory {
	DeviceLocal,
	HostStaging,
}

struct HostMapping {
	address: std::ptr::NonNull<u8>,
	length: usize,
}

/// A compute binding borrows its unique secret owner; it cannot keep storage
/// alive, map it, convert it to ordinary storage, or confirm erasure.
pub(in crate::runtime) struct SecretBinding<'a> {
	owner: &'a SecretBuffer,
	index: u32,
}

impl SecretBinding<'_> {
	pub(super) fn belongs_to(&self, device: &Device) -> bool {
		self.owner.device.same_as(device)
	}

	pub(in crate::runtime) fn descriptor_index(&self) -> u32 {
		self.index
	}

	pub(in crate::runtime) fn raw(&self) -> ash::vk::Buffer {
		self.owner.handle
	}

	pub(in crate::runtime) fn byte_len(&self) -> u64 {
		self.owner.size
	}
}

// SAFETY: dedicated VMA storage keeps the mapping stable. Its unique owner is
// transferred into command retirement before submission; CPU access occurs
// only before submission or after proven completion, never while GPU-pending.
unsafe impl Send for HostMapping {}

// SAFETY: shared references expose no host reads or writes. Mapping access is
// limited to unique staging construction and exclusive Drop after completion;
// concurrent command retention only shares device handles and atomic evidence.
unsafe impl Sync for HostMapping {}

/// Completion evidence contains no allocation, native handle, key or entropy.
pub(in crate::runtime) struct SecretErasure {
	event: Event,
	witness: ErasureWitness,
}

pub(in crate::runtime) struct ErasureWitness {
	states: Vec<Arc<AtomicU8>>,
}

impl ErasureWitness {
	pub(in crate::runtime) fn is_confirmed(&self) -> bool {
		self
			.states
			.iter()
			.all(|state| state.load(Ordering::Acquire) == CONFIRMED)
	}

	pub(in crate::runtime) fn is_unconfirmed(&self) -> bool {
		self
			.states
			.iter()
			.any(|state| state.load(Ordering::Acquire) == UNCONFIRMED)
	}
}

impl SecretBuffer {
	pub(in crate::runtime) fn new(device: &Arc<Device>, bytes: usize) -> Result<Self> {
		Self::allocate(
			device,
			bytes,
			ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST,
		)
	}

	fn allocate(
		device: &Arc<Device>,
		bytes: usize,
		usage: ash::vk::BufferUsageFlags,
	) -> Result<Self> {
		Self::allocate_with_memory(device, bytes, usage, SecretMemory::DeviceLocal)
	}

	fn allocate_with_memory(
		device: &Arc<Device>,
		bytes: usize,
		usage: ash::vk::BufferUsageFlags,
		memory: SecretMemory,
	) -> Result<Self> {
		let host = memory == SecretMemory::HostStaging;
		let size = padded_size(bytes)?;
		if !host && size > u64::from(device.physical().limits.max_storage_buffer_range) {
			return Err(Error::invalid_argument(
				"secret storage extent exceeds the device descriptor range",
			));
		}
		let info = ash::vk::BufferCreateInfo::default()
			.size(size)
			.usage(usage)
			.sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
		let allocation_info = vk_mem::AllocationCreateInfo {
			flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY
				| if host {
					vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
				} else {
					vk_mem::AllocationCreateFlags::empty()
				},
			usage: vk_mem::MemoryUsage::AutoPreferDevice,
			required_flags: if host {
				ash::vk::MemoryPropertyFlags::HOST_VISIBLE
			} else {
				ash::vk::MemoryPropertyFlags::DEVICE_LOCAL
			},
			..Default::default()
		};
		// SAFETY: this live engine device owns the allocator. Device-local secrets have
		// no host access; the separate staging allocation requests host visibility.
		// Dedicated memory avoids pools. Neither allocation requests export flags.
		let (handle, allocation) = unsafe { device.allocator().create_buffer(&info, &allocation_info) }
			.map_err(|source| Error::backend_failure("Vulkan", "secret allocation", source))?;
		let descriptor_index = if host {
			None
		} else {
			match device.bind_storage_buffer(handle, size) {
				Ok(index) => Some(index),
				Err(error) => {
					let mut allocation = allocation;
					// SAFETY: construction has not exposed, written or submitted this
					// allocation. No descriptor was acquired and no secret needs wiping.
					unsafe { device.allocator().destroy_buffer(handle, &mut allocation) };
					return Err(error);
				}
			}
		};
		Ok(Self {
			device: device.clone(),
			handle,
			allocation: Some(allocation),
			size,
			descriptor_index,
			state: Arc::new(AtomicU8::new(LIVE)),
			host_mapping: None,
			memory,
		})
	}

	pub(in crate::runtime) fn binding(&self) -> Result<SecretBinding<'_>> {
		if self.state.load(Ordering::Acquire) != LIVE {
			return Err(Error::failed_precondition("secret storage is retiring"));
		}
		let index = self
			.descriptor_index
			.ok_or_else(|| Error::failed_precondition("entropy staging is not compute storage"))?;
		Ok(SecretBinding { owner: self, index })
	}

	fn staging(device: &Arc<Device>, bytes: &[u8]) -> Result<Self> {
		if bytes.len() != 32 && bytes.len() != 64 {
			return Err(Error::invalid_argument(
				"secret entropy upload requires 32 or 64 bytes",
			));
		}
		#[cfg(not(unix))]
		return Err(Error::missing_capability(
			"locked Vulkan entropy staging requires a Unix adapter",
		));
		#[cfg(unix)]
		{
			let mut staging = Self::allocate_with_memory(
				device,
				bytes.len(),
				ash::vk::BufferUsageFlags::TRANSFER_SRC | ash::vk::BufferUsageFlags::TRANSFER_DST,
				SecretMemory::HostStaging,
			)?;
			let allocation = staging
				.allocation
				.as_mut()
				.expect("new secret staging allocation");
			// SAFETY: this unique dedicated HOST_VISIBLE allocation is not submitted.
			let address = unsafe { device.allocator().map_memory(allocation) }
				.map_err(|e| Error::backend_failure("Vulkan", "secret staging mapping", e))?;
			let Some(address) = std::ptr::NonNull::new(address) else {
				// SAFETY: release the successful mapping reference before failure.
				unsafe { device.allocator().unmap_memory(allocation) };
				return Err(Error::internal("Vulkan returned a null staging mapping"));
			};
			// SAFETY: the mapping is live and owns this complete byte range.
			if unsafe { libc::mlock(address.as_ptr().cast(), bytes.len()) } != 0 {
				let error = std::io::Error::last_os_error();
				// SAFETY: no secret was written and no command refers to the mapping.
				unsafe { device.allocator().unmap_memory(allocation) };
				return Err(Error::backend_failure(
					"host entropy",
					"staging page locking",
					error,
				));
			}
			staging.host_mapping = Some(HostMapping {
				address,
				length: bytes.len(),
			});
			// SAFETY: locked storage is uniquely owned and no GPU operation is pending.
			unsafe { std::slice::from_raw_parts_mut(address.as_ptr(), bytes.len()) }
				.copy_from_slice(bytes);
			device
				.allocator()
				.flush_allocation(allocation, 0, staging.size)
				.map_err(|e| Error::backend_failure("Vulkan", "secret staging flush", e))?;
			Ok(staging)
		}
	}

	/// Consume the allocation in a single, non-replayable upload/consumer/wipe.
	pub(in crate::runtime) fn record_entropy_consumer(
		self,
		bytes: &[u8],
		consumer: impl FnOnce(&Arc<Device>, ash::vk::CommandBuffer, SecretBinding<'_>) -> Result<()>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		if u64::try_from(bytes.len()).ok() != Some(self.size) {
			return Err(Error::invalid_argument(
				"entropy extent differs from device allocation",
			));
		}
		let staging = Self::staging(&self.device, bytes)?;
		let command = self.device.record_compute_commands(|command| {
			// SAFETY: both dedicated ranges are equal, aligned, live, disjoint and
			// carry transfer usage. Flushed host writes precede queue submission.
			unsafe {
				self.device.raw().cmd_copy_buffer(
					command,
					staging.handle,
					self.handle,
					&[ash::vk::BufferCopy::default().size(self.size)],
				);
			}
			let barrier = ash::vk::BufferMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
				.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
				.dst_access_mask(ash::vk::AccessFlags2::MEMORY_READ | ash::vk::AccessFlags2::MEMORY_WRITE)
				.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
				.buffer(self.handle)
				.size(self.size);
			// SAFETY: the copied bytes must be visible to the following consumer.
			unsafe {
				self.device.sync_commands().pipeline_barrier(
					self.device.raw(),
					command,
					&ash::vk::DependencyInfo::default()
						.buffer_memory_barriers(std::slice::from_ref(&barrier)),
				);
			}
			consumer(&self.device, command, self.binding()?)?;
			self.encode_erasure(command);
			staging.encode_erasure(command);
			let host = ash::vk::BufferMemoryBarrier2::default()
				.src_stage_mask(ash::vk::PipelineStageFlags2::CLEAR)
				.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
				.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
				.dst_access_mask(ash::vk::AccessFlags2::HOST_WRITE)
				.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
				.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
				.buffer(staging.handle)
				.size(staging.size);
			// SAFETY: after exact completion, retirement overwrites the mapped
			// host cache and flushes it before unmapping or releasing storage.
			unsafe {
				self.device.sync_commands().pipeline_barrier(
					self.device.raw(),
					command,
					&ash::vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&host)),
				);
			}
			Ok(())
		})?;
		let (command, mut witness) = self.retain_erasure(command);
		let (command, staging_witness) = staging.retain_erasure(command);
		witness.states.extend(staging_witness.states);
		Ok((command, witness))
	}

	/// Private correctness route: generate a public key and retire the private key
	/// in the same one-shot command. Long-lived key admission is a separate API.
	pub(in crate::runtime) fn record_mlkem_keygen(
		self,
		bytes: &[u8],
		private: SecretBuffer,
		public: &crate::runtime::Buffer,
		k: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let mut retention = None;
		let (mut command, mut witness) =
			self.record_entropy_consumer(bytes, |device, command, seed| {
				let prepared = device.prepare_mlkem_keygen(seed, private.binding()?, public, k)?;
				prepared.encode(command);
				retention = Some(prepared.retention());
				private.encode_erasure(command);
				Ok(())
			})?;
		let (public, sets) =
			retention.ok_or_else(|| Error::internal("ML-KEM dispatch retention missing"))?;
		command.retain_secret_public_output(public, sets)?;
		let (command, private_witness) = private.retain_erasure(command);
		witness.states.extend(private_witness.states);
		Ok((command, witness))
	}

	/// Retain the generated key for later Engine operations, erasing only entropy.
	pub(in crate::runtime) fn record_mlkem_retained_keygen(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		public: &crate::runtime::Buffer,
		k: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let mut retention = None;
		let (mut command, witness) = self.record_entropy_consumer(bytes, |device, command, seed| {
			let prepared = device.prepare_mlkem_keygen(seed, private.binding()?, public, k)?;
			prepared.encode(command);
			retention = Some(prepared.retention());
			Ok(())
		})?;
		let (public, sets) =
			retention.ok_or_else(|| Error::internal("ML-KEM dispatch retention missing"))?;
		command.retain_secret_public_output(public, sets)?;
		command.retain_secret_use(Arc::clone(private));
		Ok((command, witness))
	}

	/// Retain the generated key for later Engine operations, erasing only entropy.
	pub(in crate::runtime) fn record_mldsa_retained_keygen(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		public: &crate::runtime::Buffer,
		parameter: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let mut retention = None;
		let (mut command, witness) = self.record_entropy_consumer(bytes, |device, command, seed| {
			let prepared = device.prepare_mldsa_keygen(seed, private.binding()?, public, parameter)?;
			prepared.encode(command);
			retention = Some(prepared.retention());
			Ok(())
		})?;
		let (public, sets) =
			retention.ok_or_else(|| Error::internal("ML-DSA dispatch retention missing"))?;
		command.retain_secret_public_output(public, sets)?;
		command.retain_secret_use(Arc::clone(private));
		Ok((command, witness))
	}

	pub(in crate::runtime) fn record_mldsa_sign(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		public: [&crate::runtime::Buffer; 3],
		parameter: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		self.record_mldsa_sign_consumer(bytes, private, |device, command, seed, workspace| {
			let operands = crate::runtime::device::pqc::SecretConsumerOperands::mldsa_sign(
				device,
				[seed, private.binding()?, workspace],
				public,
				parameter,
			)?;
			let prepared = device.prepare_secret_consumer(operands)?;
			prepared.encode(command);
			Ok(prepared.retention())
		})
	}

	pub(in crate::runtime) fn record_mldsa_sign_message(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		public: [&crate::runtime::Buffer; 4],
		parameter: u32,
		lengths: [u32; 2],
		algorithm: Option<u32>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		self.record_mldsa_sign_consumer(bytes, private, |device, command, seed, workspace| {
			let operands = crate::runtime::device::pqc::SecretConsumerOperands::mldsa_sign_message(
				device,
				[seed, private.binding()?, workspace],
				public,
				parameter,
				lengths,
				algorithm,
			)?;
			let prepared = device.prepare_secret_consumer(operands)?;
			prepared.encode(command);
			Ok(prepared.retention())
		})
	}

	pub(in crate::runtime) fn record_mldsa_sign_prehashed(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		public: [&crate::runtime::Buffer; 4],
		parameter: u32,
		algorithm: u32,
		context_length: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		self.record_mldsa_sign_consumer(bytes, private, |device, command, seed, workspace| {
			let operands = crate::runtime::device::pqc::SecretConsumerOperands::mldsa_sign_prehashed(
				device,
				[seed, private.binding()?, workspace],
				public,
				parameter,
				algorithm,
				context_length,
			)?;
			let prepared = device.prepare_secret_consumer(operands)?;
			prepared.encode(command);
			Ok(prepared.retention())
		})
	}

	// One lifecycle for all signing interfaces: private scratch is never initialized
	// through host storage. Active rows are produced before use, but the full range
	// is erased on success or algorithm failure before exact retirement.
	fn record_mldsa_sign_consumer(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		consumer: impl FnOnce(
			&Arc<Device>,
			ash::vk::CommandBuffer,
			SecretBinding<'_>,
			SecretBinding<'_>,
		) -> Result<(
			Vec<crate::runtime::Buffer>,
			Option<crate::runtime::device::descriptor::BoundedSets>,
		)>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let workspace = Self::new(
			&self.device,
			crate::runtime::shader::KernelId::mldsa_sign_workspace_bytes(),
		)?;
		let mut retention = None;
		let (command, mut witness) = self.record_entropy_consumer(bytes, |device, command, seed| {
			retention = Some(consumer(device, command, seed, workspace.binding()?)?);
			workspace.encode_erasure(command);
			Ok(())
		})?;
		let (mut command, workspace_witness) = workspace.retain_erasure(command);
		witness.states.extend(workspace_witness.states);
		let (resources, sets) =
			retention.ok_or_else(|| Error::internal("ML-DSA signing retention missing"))?;
		command.retain_secret_public_resources(resources, sets)?;
		command.retain_secret_use(Arc::clone(private));
		Ok((command, witness))
	}

	pub(in crate::runtime) fn record_mlkem_encaps(
		self,
		bytes: &[u8],
		public: &crate::runtime::Buffer,
		shared: &Arc<SecretBuffer>,
		ciphertext: &crate::runtime::Buffer,
		status: &crate::runtime::Buffer,
		k: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let mut prepared_retention = None;
		let (mut command, witness) = self.record_entropy_consumer(bytes, |device, command, seed| {
			let operands = crate::runtime::device::pqc::SecretConsumerOperands::encaps(
				device,
				seed,
				public,
				shared.binding()?,
				ciphertext,
				status,
				k,
			)?;
			let prepared = device.prepare_secret_consumer(operands)?;
			prepared.encode(command);
			prepared_retention = Some(prepared.retention());
			Ok(())
		})?;
		let (public_resources, sets) =
			prepared_retention.ok_or_else(|| Error::internal("ML-KEM dispatch retention missing"))?;
		command.retain_secret_public_resources(public_resources, sets)?;
		command.retain_secret_use(Arc::clone(shared));
		Ok((command, witness))
	}

	pub(in crate::runtime) fn record_mlkem_decaps(
		self: &Arc<Self>,
		ciphertext: &crate::runtime::Buffer,
		shared: &Arc<SecretBuffer>,
		k: u32,
	) -> Result<RecordedCommandBuffer> {
		let operands = crate::runtime::device::pqc::SecretConsumerOperands::decaps(
			&self.device,
			self.binding()?,
			ciphertext,
			shared.binding()?,
			k,
		)?;
		let prepared = self.device.prepare_secret_consumer(operands)?;
		let mut command = self.device.record_compute_commands(|command| {
			prepared.encode(command);
			Ok(())
		})?;
		prepared.retain(&mut command)?;
		command.retain_secret_use(Arc::clone(self));
		command.retain_secret_use(Arc::clone(shared));
		Ok(command)
	}

	/// Consume the last value owner; older commands keep their own references.
	pub(in crate::runtime) fn record_retained_erasure(
		self: Arc<Self>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let mut command = self.device.record_compute_commands(|command| {
			self.encode_erasure(command);
			Ok(())
		})?;
		self.state.store(PENDING, Ordering::Release);
		let witness = ErasureWitness {
			states: vec![Arc::clone(&self.state)],
		};
		command.retain_shared_secret_erasure(self);
		Ok((command, witness))
	}

	pub(in crate::runtime) fn belongs_to(&self, device: &Arc<Device>) -> bool {
		self.device.same_as(device)
	}

	pub(in crate::runtime) fn record_erasure(
		self,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let command = self.device.record_compute_commands(|command| {
			self.encode_erasure(command);
			Ok(())
		})?;
		Ok(self.retain_erasure(command))
	}

	fn retain_erasure(
		self,
		mut command: RecordedCommandBuffer,
	) -> (RecordedCommandBuffer, ErasureWitness) {
		self.state.store(PENDING, Ordering::Release);
		let witness = ErasureWitness {
			states: vec![Arc::clone(&self.state)],
		};
		// Retain the unique owner before submission, independently of its ticket.
		command.retain_secret_erasure(self);
		(command, witness)
	}

	fn encode_erasure(&self, command: ash::vk::CommandBuffer) {
		let barrier = ash::vk::BufferMemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::ALL_COMMANDS)
			.src_access_mask(ash::vk::AccessFlags2::MEMORY_READ | ash::vk::AccessFlags2::MEMORY_WRITE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::CLEAR)
			.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
			.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
			.buffer(self.handle)
			.offset(0)
			.size(self.size);
		// SAFETY: Engine submits this compute-only allocation after the final
		// consumer on its ordered timeline. The barrier covers the entire range,
		// including padding. No family transfer or protected memory is involved.
		unsafe {
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&ash::vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&barrier)),
			);
			self
				.device
				.raw()
				.cmd_fill_buffer(command, self.handle, 0, self.size, 0);
		}
	}

	pub(super) fn confirm_erasure(&self) {
		// Mapped staging additionally requires CPU-cache erasure and flush.
		// Its witness stays incomplete until Drop finishes that retirement work.
		let state = if self.memory == SecretMemory::HostStaging {
			GPU_COMPLETE
		} else {
			CONFIRMED
		};
		self.state.store(state, Ordering::Release);
	}

	pub(super) fn fail_erasure(&self) {
		self.state.store(UNCONFIRMED, Ordering::Release);
	}
}

impl SecretErasure {
	pub(in crate::runtime) fn new(event: Event, witness: ErasureWitness) -> Self {
		Self { event, witness }
	}

	pub(in crate::runtime) fn try_is_complete(&self) -> Result<bool> {
		if self.witness.is_unconfirmed() {
			return Err(Error::failed_precondition("secret erasure is unconfirmed"));
		}
		Ok(self.event.is_complete()? && self.witness.is_confirmed())
	}

	pub(in crate::runtime) fn wait(&self) -> Result<()> {
		self.event.wait()?;
		if !self.witness.is_confirmed() {
			return Err(Error::failed_precondition("secret erasure is unconfirmed"));
		}
		Ok(())
	}
}

impl Drop for SecretBuffer {
	fn drop(&mut self) {
		let Some(mut allocation) = self.allocation.take() else {
			return;
		};
		let state = self.state.load(Ordering::Acquire);
		let can_release_staging =
			self.memory == SecretMemory::HostStaging && (state == LIVE || state == GPU_COMPLETE);
		if let Some(mapping) = self.host_mapping.as_ref().filter(|_| can_release_staging) {
			// SAFETY: LIVE staging was never submitted; GPU_COMPLETE staging has
			// finished every GPU access. This unique mapping remains locked/live.
			crate::core::memory::zero_secure(unsafe {
				std::slice::from_raw_parts_mut(mapping.address.as_ptr(), mapping.length)
			});
			if self
				.device
				.allocator()
				.flush_allocation(&allocation, 0, self.size)
				.is_err()
			{
				self.state.store(UNCONFIRMED, Ordering::Release);
				mem::forget(self.device.clone());
				return;
			}
			#[cfg(unix)]
			// SAFETY: this is the exact live range previously locked by staging.
			unsafe {
				libc::munlock(mapping.address.as_ptr().cast(), mapping.length);
			}
			// SAFETY: CPU erasure is flushed and no GPU access remains.
			unsafe {
				self.device.allocator().unmap_memory(&mut allocation);
			}
		}
		if state != CONFIRMED && !can_release_staging {
			self.state.store(UNCONFIRMED, Ordering::Release);
			// Terminal quarantine: neither VMA nor ordinary storage can reuse the
			// range. Retain its allocator/device forever rather than claim erasure.
			// Drop records no commands, reads no bytes and performs no wait.
			// VMA's allocation is a Copy handle with no destructor. Deliberately
			// omit destroy_buffer; the leaked Arc<Device> keeps its dedicated memory live.
			mem::forget(self.device.clone());
			return;
		}
		// SAFETY: successful retirement proves completion, or LIVE host staging
		// was never submitted. Its host erasure is flushed before reclamation.
		unsafe {
			self
				.device
				.allocator()
				.destroy_buffer(self.handle, &mut allocation);
		}
		if let Some(index) = self.descriptor_index.take() {
			self.device.release_storage_buffer(index);
		}
		if self.memory == SecretMemory::HostStaging && state == GPU_COMPLETE {
			self.state.store(CONFIRMED, Ordering::Release);
		}
	}
}

fn padded_size(bytes: usize) -> Result<u64> {
	if bytes == 0 {
		return Err(Error::invalid_argument(
			"secret allocation must not be empty",
		));
	}
	let padded = bytes
		.checked_add(3)
		.map(|size| size & !3)
		.ok_or_else(|| Error::invalid_argument("secret allocation size overflows"))?;
	u64::try_from(padded)
		.map_err(|_| Error::invalid_argument("secret allocation exceeds device size"))
}

#[cfg(test)]
#[path = "../../../../test/rs/runtime/secret_buffer_unit.rs"]
mod tests;
