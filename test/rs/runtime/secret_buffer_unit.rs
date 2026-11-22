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
/// Only this test-owned type can observe a fixture allocation. Its private
/// field prevents constructing it from a production secret owner.
pub(in crate::runtime) struct FixtureSecret {
	secret: Arc<SecretBuffer>,
}

impl FixtureSecret {
	pub(in crate::runtime) fn owner(&self) -> &Arc<SecretBuffer> {
		&self.secret
	}

	pub(in crate::runtime) fn into_owner(self) -> Arc<SecretBuffer> {
		self.secret
	}

	/// Disposable fixture allocations only: the shipping constructor never adds
	/// TRANSFER_SRC. Callers retain the public probe through the returned event.
	pub(in crate::runtime) fn test_record_fixture_snapshot(
		&self,
		probe: &crate::runtime::Buffer,
	) -> Result<RecordedCommandBuffer> {
		let secret = &self.secret;
		assert!(probe.belongs_to(&secret.device));
		assert_eq!(probe.size(), secret.size);
		let mut command = secret.device.record_compute_commands(|command| {
			let ready = vk::BufferMemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
				.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
				.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
				.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
				.set_buffer(secret.handle)
				.set_size(secret.size);
			let host = vk::MemoryBarrier2::new()
				.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
				.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
				.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
				.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT);
			// SAFETY: exclusively test_new fixture storage supplies TRANSFER_SRC;
			// equal live ranges and barriers cover preceding device writes and host reads.
			unsafe {
				secret.device.sync_commands().pipeline_barrier(
					secret.device.raw(),
					command,
					&vk::DependencyInfo::new()
						.set_buffer_memory_barriers(std::slice::from_ref(&ready))
						.map_err(Error::from)?,
				)?;
				secret.device.raw().cmd_copy_buffer(
					command,
					secret.handle,
					probe.raw(),
					&[vk::BufferCopy::new().set_size(secret.size)],
				)?;
				secret.device.sync_commands().pipeline_barrier(
					secret.device.raw(),
					command,
					&vk::DependencyInfo::new()
						.set_memory_barriers(std::slice::from_ref(&host))
						.map_err(Error::from)?,
				)?;
			}
			Ok(())
		})?;
		command.retain_secret_use(Arc::clone(secret));
		command.retain_secret_public_resources(std::iter::once(probe.clone()), None)?;
		Ok(command)
	}
}

impl SecretBuffer {
	pub(in crate::runtime) fn test_fixture(device: &Arc<Device>) -> Result<FixtureSecret> {
		Ok(FixtureSecret {
			secret: Arc::new(Self::test_new(device, 32)?),
		})
	}

	pub(in crate::runtime) fn test_new(device: &Arc<Device>, bytes: usize) -> Result<Self> {
		Self::allocate(
			device,
			bytes,
			vk::BufferUsageFlagBits::STORAGE_BUFFER_BIT
				| vk::BufferUsageFlagBits::TRANSFER_DST_BIT
				| vk::BufferUsageFlagBits::TRANSFER_SRC_BIT,
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
					.cmd_fill_buffer(command, self.handle, 0, self.size, 0xa5a5a5a5)?;
			}
			self.encode_erasure(command)?;
			self.record_test_readback(command, probe)?;
			Ok(())
		})?;
		Ok(self.retain_erasure(command))
	}
	fn record_test_readback(
		&self,
		command: vk::CommandBuffer,
		probe: &crate::runtime::Buffer,
	) -> Result<()> {
		let barrier = vk::BufferMemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::CLEAR_BIT)
			.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::TRANSFER_READ_BIT)
			.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.set_buffer(self.handle)
			.set_size(self.size);
		let host = vk::MemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
			.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT);
		// SAFETY: the secret is command-retained and the test keeps the probe
		// alive through completion. Both buffers have the
		// required transfer usage and exact equal ranges. No production secrets.
		unsafe {
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&vk::DependencyInfo::new()
					.set_buffer_memory_barriers(std::slice::from_ref(&barrier))
					.map_err(Error::from)?,
			)?;
			self.device.raw().cmd_copy_buffer(
				command,
				self.handle,
				probe.raw(),
				&[vk::BufferCopy::new().set_size(self.size)],
			)?;
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&vk::DependencyInfo::new()
					.set_memory_barriers(std::slice::from_ref(&host))
					.map_err(Error::from)?,
			)?;
		}
		Ok(())
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
					&[vk::BufferCopy::new().set_size(probe.size())],
				)?;
				let host = vk::MemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COPY_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::TRANSFER_WRITE_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
					.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT);
				device.sync_commands().pipeline_barrier(
					device.raw(),
					command,
					&vk::DependencyInfo::new()
						.set_memory_barriers(std::slice::from_ref(&host))
						.map_err(Error::from)?,
				)?;
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
