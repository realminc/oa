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

use crate::{Error, Result, runtime::Event};

use crate::runtime::{Device, RecordedCommandBuffer};

const LIVE: u8 = 0;
const PENDING: u8 = 1;
const CONFIRMED: u8 = 2;
const UNCONFIRMED: u8 = 3;
const GPU_COMPLETE: u8 = 4;

pub(in crate::runtime) struct SecretBuffer {
	device: Arc<Device>,
	handle: vk::Buffer,
	allocation: Option<vk::mem::Allocation>,
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

	pub(in crate::runtime) fn raw(&self) -> vk::Buffer {
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
pub(crate) struct SecretErasure {
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
			vk::BufferUsageFlagBits::STORAGE_BUFFER_BIT | vk::BufferUsageFlagBits::TRANSFER_DST_BIT,
		)
	}

	fn allocate(device: &Arc<Device>, bytes: usize, usage: vk::BufferUsageFlags) -> Result<Self> {
		Self::allocate_with_memory(device, bytes, usage, SecretMemory::DeviceLocal)
	}

	fn allocate_with_memory(
		device: &Arc<Device>,
		bytes: usize,
		usage: vk::BufferUsageFlags,
		memory: SecretMemory,
	) -> Result<Self> {
		let host = memory == SecretMemory::HostStaging;
		let size = padded_size(bytes)?;
		if !host && size > u64::from(device.physical().limits.max_storage_buffer_range) {
			return Err(Error::invalid_argument(
				"secret storage extent exceeds the device descriptor range",
			));
		}
		let info = vk::BufferCreateInfo::new()
			.set_size(size)
			.set_usage(usage)
			.set_sharing_mode(vk::SharingMode::EXCLUSIVE);
		let allocation_info = vk::mem::AllocationCreateInfo {
			dedicated_memory: true,
			host_access: if host {
				vk::mem::HostAccess::SequentialWrite
			} else {
				vk::mem::HostAccess::None
			},
			usage: vk::mem::MemoryUsage::PreferDevice,
			required_flags: if host {
				vk::MemoryPropertyFlagBits::HOST_VISIBLE_BIT
			} else {
				vk::MemoryPropertyFlagBits::DEVICE_LOCAL_BIT
			},
			..Default::default()
		};
		// SAFETY: this live engine device owns the allocator. Device-local secrets have
		// no host access; the separate staging allocation requests host visibility.
		// Dedicated memory avoids pools. Neither allocation requests export flags.
		let (handle, allocation) = unsafe {
			device
				.allocator()
				.with_allocator(|a| a.create_buffer_with_options(&info, &allocation_info))
		}
		.map_err(|source| Error::backend_failure("Vulkan", "secret allocation", source))?
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
					unsafe {
						device
							.allocator()
							.with_allocator(|a| a.destroy_buffer(handle, &mut allocation))
					}??;
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
		if bytes.is_empty() {
			return Err(Error::invalid_argument(
				"secret upload requires non-empty bytes",
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
				vk::BufferUsageFlagBits::TRANSFER_SRC_BIT | vk::BufferUsageFlagBits::TRANSFER_DST_BIT,
				SecretMemory::HostStaging,
			)?;
			let allocation = staging
				.allocation
				.as_mut()
				.expect("new secret staging allocation");
			// SAFETY: this unique dedicated HOST_VISIBLE allocation is not submitted.
			let address = unsafe { device.allocator().with_allocator(|a| a.map(allocation)) }
				.map_err(|e| Error::backend_failure("Vulkan", "secret staging mapping", e))?
				.map_err(Error::from)?;
			let Some(address) = std::ptr::NonNull::new(address as *mut u8) else {
				// SAFETY: release the successful mapping reference before failure.
				unsafe { device.allocator().with_allocator(|a| a.unmap(allocation)) }??;
				return Err(Error::internal("Vulkan returned a null staging mapping"));
			};
			// SAFETY: the mapping is live and owns this complete byte range.
			if unsafe { libc::mlock(address.as_ptr().cast(), bytes.len()) } != 0 {
				let error = std::io::Error::last_os_error();
				// SAFETY: no secret was written and no command refers to the mapping.
				unsafe { device.allocator().with_allocator(|a| a.unmap(allocation)) }??;
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
			// SAFETY: `allocation` is live, uniquely owned by this staging owner,
			// and its mapped range covers `staging.size` bytes.
			unsafe {
				device
					.allocator()
					.with_allocator(|a| a.flush(allocation, 0, staging.size))
			}
			.map_err(|e| Error::backend_failure("Vulkan", "secret staging flush", e))?
			.map_err(|e| Error::backend_failure("Vulkan", "secret staging flush", e))?;
			Ok(staging)
		}
	}

	/// Consume the allocation in a single, non-replayable upload/consumer/wipe.
	pub(in crate::runtime) fn record_entropy_consumer(
		self,
		bytes: &[u8],
		consumer: impl FnOnce(&Arc<Device>, vk::CommandBuffer, SecretBinding<'_>) -> Result<()>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		if bytes.len() != 32 && bytes.len() != 64 {
			return Err(Error::invalid_argument(
				"secret entropy upload requires 32 or 64 bytes",
			));
		}
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
					&[vk::BufferCopy::new().set_size(self.size)],
				)?;
			}
			let barrier = vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
				.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
				.set_dst_access_mask(
					vk::AccessFlagBits2::MEMORY_READ_BIT | vk::AccessFlagBits2::MEMORY_WRITE_BIT,
				)
				.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_buffer(self.handle)
				.set_size(self.size);
			// SAFETY: the copied bytes must be visible to the following consumer.
			unsafe {
				self
					.device
					.sync_commands()
					.pipeline_barrier(
						self.device.raw(),
						command,
						&(vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&barrier)))
							.map_err(Error::from)?,
					)
					.map_err(Error::from)?;
			}
			consumer(&self.device, command, self.binding()?)?;
			self.encode_erasure(command)?;
			staging.encode_erasure(command)?;
			let host = vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::CLEAR_BIT)
				.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
				.set_dst_access_mask(vk::AccessFlagBits2::HOST_WRITE_BIT)
				.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_buffer(staging.handle)
				.set_size(staging.size);
			// SAFETY: after exact completion, retirement overwrites the mapped
			// host cache and flushes it before unmapping or releasing storage.
			unsafe {
				self
					.device
					.sync_commands()
					.pipeline_barrier(
						self.device.raw(),
						command,
						&(vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&host)))
							.map_err(Error::from)?,
					)
					.map_err(Error::from)?;
			}
			Ok(())
		})?;
		let (command, mut witness) = self.retain_erasure(command);
		let (command, staging_witness) = staging.retain_erasure(command);
		witness.states.extend(staging_witness.states);
		Ok((command, witness))
	}

	/// Upload host secret bytes into retained device storage through locked
	/// staging. Only the staging copy is erased; the destination stays LIVE and
	/// is retained through the submission. `after` encodes optional follow-up
	/// work ordered behind the upload barrier (for example a public-key copy).
	pub(in crate::runtime) fn record_secret_import(
		self: &Arc<Self>,
		bytes: &[u8],
		after: impl FnOnce(&Arc<Device>, vk::CommandBuffer) -> Result<()>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		if u64::try_from(bytes.len()).ok() != Some(self.size) {
			return Err(Error::invalid_argument(
				"secret import extent differs from device allocation",
			));
		}
		if self.memory != SecretMemory::DeviceLocal {
			return Err(Error::invalid_argument(
				"secret import requires device-local storage",
			));
		}
		if self.state.load(Ordering::Acquire) != LIVE {
			return Err(Error::failed_precondition(
				"secret import requires live storage",
			));
		}
		let staging = Self::staging(&self.device, bytes)?;
		let mut command = self.device.record_compute_commands(|command| {
			// SAFETY: both dedicated ranges are equal, aligned, live, disjoint and
			// carry transfer usage. Flushed host writes precede queue submission.
			unsafe {
				self.device.raw().cmd_copy_buffer(
					command,
					staging.handle,
					self.handle,
					&[vk::BufferCopy::new().set_size(self.size)],
				)?;
			}
			let barrier = vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
				.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
				.set_dst_access_mask(
					vk::AccessFlagBits2::MEMORY_READ_BIT | vk::AccessFlagBits2::MEMORY_WRITE_BIT,
				)
				.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_buffer(self.handle)
				.set_size(self.size);
			// SAFETY: the imported bytes must be visible to every later consumer.
			unsafe {
				self
					.device
					.sync_commands()
					.pipeline_barrier(
						self.device.raw(),
						command,
						&(vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&barrier)))
							.map_err(Error::from)?,
					)
					.map_err(Error::from)?;
			}
			after(&self.device, command)?;
			staging.encode_erasure(command)?;
			let host = vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::CLEAR_BIT)
				.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
				.set_dst_access_mask(vk::AccessFlagBits2::HOST_WRITE_BIT)
				.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_buffer(staging.handle)
				.set_size(staging.size);
			// SAFETY: after exact completion, retirement overwrites the mapped
			// host cache and flushes it before unmapping or releasing storage.
			unsafe {
				self
					.device
					.sync_commands()
					.pipeline_barrier(
						self.device.raw(),
						command,
						&(vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&host)))
							.map_err(Error::from)?,
					)
					.map_err(Error::from)?;
			}
			Ok(())
		})?;
		command.retain_secret_use(Arc::clone(self));
		let (command, witness) = staging.retain_erasure(command);
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
		let workspace = Self::new(
			&self.device,
			crate::runtime::dna::kernel::KernelId::mlkem_keygen_workspace_bytes(),
		)?;
		let mut retention = None;
		let (mut command, mut witness) =
			self.record_entropy_consumer(bytes, |device, command, seed| {
				let prepared =
					device.prepare_mlkem_keygen(seed, private.binding()?, public, workspace.binding()?, k)?;
				prepared.encode(command)?;
				retention = Some(prepared.retention());
				workspace.encode_erasure(command)?;
				private.encode_erasure(command)?;
				Ok(())
			})?;
		let (next_command, workspace_witness) = workspace.retain_erasure(command);
		command = next_command;
		witness.states.extend(workspace_witness.states);
		let (public, sets) =
			retention.ok_or_else(|| Error::internal("ML-KEM dispatch retention missing"))?;
		command.retain_secret_public_output(public, sets)?;
		let (command, private_witness) = private.retain_erasure(command);
		witness.states.extend(private_witness.states);
		Ok((command, witness))
	}

	/// Retain the generated key; erase entropy and the polynomial workspace.
	pub(in crate::runtime) fn record_mlkem_retained_keygen(
		self,
		bytes: &[u8],
		private: &Arc<SecretBuffer>,
		public: &crate::runtime::Buffer,
		k: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let workspace = Self::new(
			&self.device,
			crate::runtime::dna::kernel::KernelId::mlkem_keygen_workspace_bytes(),
		)?;
		let mut retention = None;
		let (mut command, mut witness) =
			self.record_entropy_consumer(bytes, |device, command, seed| {
				let prepared =
					device.prepare_mlkem_keygen(seed, private.binding()?, public, workspace.binding()?, k)?;
				prepared.encode(command)?;
				retention = Some(prepared.retention());
				workspace.encode_erasure(command)?;
				Ok(())
			})?;
		let (next_command, workspace_witness) = workspace.retain_erasure(command);
		command = next_command;
		witness.states.extend(workspace_witness.states);
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
			prepared.encode(command)?;
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
			prepared.encode(command)?;
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
			prepared.encode(command)?;
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
			prepared.encode(command)?;
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
			vk::CommandBuffer,
			SecretBinding<'_>,
			SecretBinding<'_>,
		) -> Result<(
			Vec<crate::runtime::Buffer>,
			Option<crate::runtime::device::descriptor::BoundedSets>,
		)>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let workspace = Self::new(
			&self.device,
			crate::runtime::dna::kernel::KernelId::mldsa_sign_workspace_bytes(),
		)?;
		let mut retention = None;
		let (command, mut witness) = self.record_entropy_consumer(bytes, |device, command, seed| {
			retention = Some(consumer(device, command, seed, workspace.binding()?)?);
			workspace.encode_erasure(command)?;
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
		let workspace = Self::new(
			&self.device,
			crate::runtime::dna::kernel::KernelId::mlkem_encaps_workspace_bytes(),
		)?;
		let mut prepared_retention = None;
		let (command, mut witness) = self.record_entropy_consumer(bytes, |device, command, seed| {
			let operands = crate::runtime::device::pqc::SecretConsumerOperands::encaps(
				device,
				[seed, shared.binding()?, workspace.binding()?],
				public,
				ciphertext,
				status,
				k,
			)?;
			let prepared = device.prepare_secret_consumer(operands)?;
			prepared.encode(command)?;
			workspace.encode_erasure(command)?;
			prepared_retention = Some(prepared.retention());
			Ok(())
		})?;
		let (mut command, workspace_witness) = workspace.retain_erasure(command);
		witness.states.extend(workspace_witness.states);
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
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let workspace = Self::new(
			&self.device,
			crate::runtime::dna::kernel::KernelId::mlkem_decaps_workspace_bytes(),
		)?;
		let operands = crate::runtime::device::pqc::SecretConsumerOperands::decaps(
			&self.device,
			[self.binding()?, shared.binding()?, workspace.binding()?],
			ciphertext,
			k,
		)?;
		let prepared = self.device.prepare_secret_consumer(operands)?;
		let mut command = self.device.record_compute_commands(|command| {
			prepared.encode(command)?;
			workspace.encode_erasure(command)?;
			Ok(())
		})?;
		prepared.retain(&mut command)?;
		let (mut command, witness) = workspace.retain_erasure(command);
		command.retain_secret_use(Arc::clone(self));
		command.retain_secret_use(Arc::clone(shared));
		Ok((command, witness))
	}

	/// Consumes the shared secret inside a keyed-hash declassification
	/// boundary: public info is absorbed on device and only derived public
	/// bytes leave private storage. No workspace exists to erase; the empty
	/// witness confirms trivially.
	pub(in crate::runtime) fn record_shared_derive(
		self: &Arc<Self>,
		info: &crate::runtime::Buffer,
		output: &crate::runtime::Buffer,
		info_len: u32,
		output_len: u32,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let operands = crate::runtime::device::pqc::SecretConsumerOperands::shared_derive(
			&self.device,
			self.binding()?,
			info,
			output,
			info_len,
			output_len,
		)?;
		let prepared = self.device.prepare_secret_consumer(operands)?;
		let mut command = self.device.record_compute_commands(|command| {
			prepared.encode(command)?;
			Ok(())
		})?;
		prepared.retain(&mut command)?;
		command.retain_secret_use(Arc::clone(self));
		Ok((command, ErasureWitness { states: Vec::new() }))
	}

	/// Consume the last value owner; older commands keep their own references.
	pub(in crate::runtime) fn record_retained_erasure(
		self: Arc<Self>,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		let mut command = self.device.record_compute_commands(|command| {
			self.encode_erasure(command)?;
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
			self.encode_erasure(command)?;
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

	fn encode_erasure(&self, command: vk::CommandBuffer) -> Result<()> {
		let barrier = vk::BufferMemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT)
			.set_src_access_mask(
				vk::AccessFlagBits2::MEMORY_READ_BIT | vk::AccessFlagBits2::MEMORY_WRITE_BIT,
			)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::CLEAR_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
			.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.set_buffer(self.handle)
			.set_offset(0)
			.set_size(self.size);
		// SAFETY: Engine submits this compute-only allocation after the final
		// consumer on its ordered timeline. The barrier covers the entire range,
		// including padding. No family transfer or protected memory is involved.
		unsafe {
			self
				.device
				.sync_commands()
				.pipeline_barrier(
					self.device.raw(),
					command,
					&(vk::DependencyInfo::new().set_buffer_memory_barriers(std::slice::from_ref(&barrier)))
						.map_err(Error::from)?,
				)
				.map_err(Error::from)?;
			self
				.device
				.raw()
				.cmd_fill_buffer(command, self.handle, 0, self.size, 0)
				.map_err(Error::from)?;
		}
		Ok(())
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
	pub(in crate::runtime) fn event(&self) -> &Event {
		&self.event
	}
	pub(in crate::runtime) fn new(event: Event, witness: ErasureWitness) -> Self {
		Self { event, witness }
	}

	pub(crate) fn try_is_complete(&self) -> Result<bool> {
		if self.witness.is_unconfirmed() {
			return Err(Error::failed_precondition("secret erasure is unconfirmed"));
		}
		Ok(self.event.is_complete()? && self.witness.is_confirmed())
	}

	pub(crate) fn wait(&self) -> Result<()> {
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
			if unsafe {
				self
					.device
					.allocator()
					.with_allocator(|a| a.flush(&allocation, 0, self.size))
			}
			.map_or(true, |result| result.is_err())
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
				let _ = self
					.device
					.allocator()
					.with_allocator(|a| a.unmap(&allocation));
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
			let _ = self
				.device
				.allocator()
				.with_allocator(|a| a.destroy_buffer(self.handle, &mut allocation));
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
