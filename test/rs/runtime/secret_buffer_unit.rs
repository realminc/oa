use std::sync::Arc;

use super::*;

#[test]
fn erasure_extent_rejects_empty_and_overflow_and_covers_padding() {
	assert!(padded_size(0).is_err());
	assert!(padded_size(usize::MAX).is_err());
	assert!(padded_size(usize::MAX - 1).is_err());
	for (logical, physical) in [(1, 4), (3, 4), (4, 4), (19, 20), (4097, 4100)] {
		assert_eq!(padded_size(logical).unwrap(), physical);
	}
}

#[test]
fn signing_erasure_witness_requires_entropy_staging_and_workspace() {
	let states: Vec<_> = (0..3).map(|_| Arc::new(AtomicU8::new(PENDING))).collect();
	let witness = ErasureWitness {
		states: states.clone(),
	};
	for state in &states[..2] {
		state.store(CONFIRMED, Ordering::Release);
	}
	assert!(
		!witness.is_confirmed(),
		"unwiped workspace cannot retire with entropy"
	);
	assert!(!witness.is_unconfirmed());
	states[2].store(UNCONFIRMED, Ordering::Release);
	assert!(
		witness.is_unconfirmed(),
		"workspace failure fails the combined ticket"
	);
	assert!(!witness.is_confirmed());
	// This direct transition is disposable test metadata, not a production retry.
	states[2].store(CONFIRMED, Ordering::Release);
	assert!(witness.is_confirmed());
	assert!(!witness.is_unconfirmed());
}

// Disposable instrumentation lives only in test/rs. The shipping constructor
// has no transfer-source usage or native-resource exposure.
impl SecretBuffer {
	pub(in crate::runtime) fn test_new(device: &Arc<Device>, bytes: usize) -> Result<Self> {
		Self::allocate(
			device,
			bytes,
			ash::vk::BufferUsageFlags::STORAGE_BUFFER
				| ash::vk::BufferUsageFlags::TRANSFER_DST
				| ash::vk::BufferUsageFlags::TRANSFER_SRC,
		)
	}

	pub(in crate::runtime) fn test_record_erasure(
		self,
		probe: &crate::runtime::Buffer,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		assert_eq!(probe.size(), self.size);
		let command = self.device.record_compute_commands(|command| {
			// SAFETY: this disposable buffer owns a full aligned TRANSFER_DST range.
			unsafe {
				self
					.device
					.raw()
					.cmd_fill_buffer(command, self.handle, 0, self.size, 0xa5a5a5a5);
			}
			self.encode_erasure(command);
			self.record_test_readback(command, probe);
			Ok(())
		})?;
		Ok(self.retain_erasure(command))
	}
	fn record_test_readback(&self, command: ash::vk::CommandBuffer, probe: &crate::runtime::Buffer) {
		let barrier = ash::vk::BufferMemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::CLEAR)
			.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::COPY)
			.dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
			.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
			.buffer(self.handle)
			.size(self.size);
		let host = ash::vk::MemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
			.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
			.dst_access_mask(ash::vk::AccessFlags2::HOST_READ);
		// SAFETY: the secret is command-retained and the test keeps the probe
		// alive through completion. Both buffers have the
		// required transfer usage and exact equal ranges. No production secrets.
		unsafe {
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&ash::vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&barrier)),
			);
			self.device.raw().cmd_copy_buffer(
				command,
				self.handle,
				probe.raw(),
				&[ash::vk::BufferCopy::default().size(self.size)],
			);
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&ash::vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&host)),
			);
		}
	}
}

impl SecretBuffer {
	pub(in crate::runtime) fn test_record_entropy_readback(
		self,
		bytes: &[u8],
		probe: &crate::runtime::Buffer,
	) -> Result<(RecordedCommandBuffer, ErasureWitness)> {
		self.record_entropy_consumer(bytes, |device, command, input| {
			assert_eq!(input.byte_len(), probe.size());
			// SAFETY: test-owned probe and input have equal transfer-capable ranges;
			// upload visibility is provided by the production consumer barrier.
			unsafe {
				device.raw().cmd_copy_buffer(
					command,
					input.raw(),
					probe.raw(),
					&[ash::vk::BufferCopy::default().size(probe.size())],
				);
				let host = ash::vk::MemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
					.src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
					.dst_access_mask(ash::vk::AccessFlags2::HOST_READ);
				device.sync_commands().pipeline_barrier(
					device.raw(),
					command,
					&ash::vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&host)),
				);
			}
			Ok(())
		})
	}
}

impl SecretBuffer {
	pub(in crate::runtime) fn test_staging_witness(device: &Arc<Device>) -> Result<()> {
		let staging = Self::staging(device, &[0xa5; 32])?;
		assert!(staging.binding().is_err());
		let witness = ErasureWitness {
			states: vec![Arc::clone(&staging.state)],
		};
		// Disposable retirement simulation: GPU completion alone is insufficient.
		staging.confirm_erasure();
		assert!(staging.binding().is_err());
		assert!(!witness.is_confirmed());
		assert!(!witness.is_unconfirmed());
		drop(staging);
		assert!(witness.is_confirmed());
		Ok(())
	}
}

// Observe retention metadata only; test instrumentation cannot map secret bytes.
impl SecretErasure {
	pub(in crate::runtime) fn test_retained_allocation_count(&self) -> usize {
		self.witness.states.len()
	}
}
