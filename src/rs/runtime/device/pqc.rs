//! Private PQC recording through the Engine-owned pipeline registry.
//! FIPS 203/204 formulas are imported unchanged; C++ oaPqc.md has no live donor.

use std::sync::Arc;

use super::{
	descriptor::BoundedSets, features::ExecutionProfile, pipeline::ComputePipeline,
	secret_buffer::SecretBinding,
};
use crate::runtime::{Buffer, Device};
use crate::{Error, Result, runtime::shader::KernelId};

/// Constructed only by the algorithm-specific checked constructors below.
struct SecretKeygenLayout {
	kernel: KernelId,
	parameter: u32,
	seed_len: u64,
	public_len: u64,
	private_len: u64,
}

/// Checked single-dispatch value. Borrowed secret bindings keep both owners live
/// while encoding; the enclosing one-shot recorder transfers them to erasure
/// retention before submission. No ordinary buffer can occupy a secret operand.
pub(in crate::runtime) struct PreparedSecretKeygen<'a> {
	device: &'a Arc<Device>,
	pipeline: &'a ComputePipeline,
	_seed: SecretBinding<'a>,
	_private: SecretBinding<'a>,
	public: Buffer,
	sets: Option<BoundedSets>,
	push: [u8; 16],
}

/// Ownership and wire admission precede fallible pipeline creation.
pub(super) struct SecretKeygenOperands<'a> {
	device: &'a Arc<Device>,
	seed: SecretBinding<'a>,
	private: SecretBinding<'a>,
	public: Buffer,
	parameter: u32,
	kernel: KernelId,
	seed_len: u64,
	el: u64,
	dl: u64,
}

impl<'a> SecretKeygenOperands<'a> {
	pub(super) fn mlkem(
		device: &'a Arc<Device>,
		seed: SecretBinding<'a>,
		private: SecretBinding<'a>,
		public: &Buffer,
		k: u32,
	) -> Result<Self> {
		if !(2..=4).contains(&k) {
			return Err(Error::invalid_argument("ML-KEM k must be 2, 3 or 4"));
		}
		Self::new(
			device,
			seed,
			private,
			public,
			SecretKeygenLayout {
				kernel: KernelId::CryptographyMlKemKeygenU8,
				parameter: k,
				seed_len: 64,
				public_len: u64::from(384 * k + 32),
				private_len: u64::from(768 * k + 96),
			},
		)
	}

	pub(super) fn mldsa(
		device: &'a Arc<Device>,
		seed: SecretBinding<'a>,
		private: SecretBinding<'a>,
		public: &Buffer,
		parameter: u32,
	) -> Result<Self> {
		let (seed_len, public_len, private_len) = KernelId::mldsa_keygen_layout(parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		Self::new(
			device,
			seed,
			private,
			public,
			SecretKeygenLayout {
				kernel: KernelId::CryptographyMlDsaKeygenU8,
				parameter,
				seed_len,
				public_len,
				private_len,
			},
		)
	}

	fn new(
		device: &'a Arc<Device>,
		seed: SecretBinding<'a>,
		private: SecretBinding<'a>,
		public: &Buffer,
		layout: SecretKeygenLayout,
	) -> Result<Self> {
		let SecretKeygenLayout {
			kernel,
			parameter,
			seed_len,
			public_len: el,
			private_len: dl,
		} = layout;
		if !seed.belongs_to(device) || !private.belongs_to(device) || !public.belongs_to(device) {
			return Err(Error::invalid_argument(
				"PQC operands require their originating Engine",
			));
		}
		if seed.raw() == private.raw() || seed.raw() == public.raw() || private.raw() == public.raw() {
			return Err(Error::invalid_argument(
				"PQC keygen operands must be disjoint",
			));
		}
		if seed.byte_len() != seed_len || private.byte_len() != dl || public.size() != el {
			return Err(Error::invalid_argument(
				"PQC keygen operand extent mismatch",
			));
		}
		Ok(Self {
			device,
			seed,
			private,
			public: public.clone(),
			parameter,
			kernel,
			seed_len,
			el,
			dl,
		})
	}

	pub(super) fn kernel(&self) -> KernelId {
		self.kernel
	}

	pub(super) fn prepare(self, pipeline: &'a ComputePipeline) -> Result<PreparedSecretKeygen<'a>> {
		let Self {
			device,
			seed,
			private,
			public,
			parameter,
			kernel,
			seed_len,
			el,
			dl,
		} = self;
		if pipeline.push_constant_size() != 16 || pipeline.max_dispatch_group_count().contains(&0) {
			return Err(Error::missing_capability("PQC dispatch ABI unavailable"));
		}
		let bounded = device.physical().profile == ExecutionProfile::Compatibility;
		let sets = if bounded {
			let layout = device
				.bounded_descriptor_layout(kernel.bounded_buffer_count())
				.ok_or_else(|| Error::missing_capability("PQC bounded descriptor layout unavailable"))?;
			let buffers = vec![
				ash::vk::DescriptorBufferInfo::default()
					.buffer(seed.raw())
					.range(seed_len),
				ash::vk::DescriptorBufferInfo::default()
					.buffer(private.raw())
					.range(dl),
				ash::vk::DescriptorBufferInfo::default()
					.buffer(public.raw())
					.range(el),
			];
			// BoundedSets retains the Device and descriptors until command retirement.
			Some(BoundedSets::new(device, &[layout], &[buffers])?)
		} else {
			None
		};
		let indices = if bounded {
			[0, 1, 2]
		} else {
			[
				seed.descriptor_index(),
				private.descriptor_index(),
				public.descriptor_index(),
			]
		};
		let mut push = [0; 16];
		for (bytes, value) in push
			.as_chunks_mut::<4>()
			.0
			.iter_mut()
			.zip(indices.into_iter().chain([parameter]))
		{
			bytes.copy_from_slice(&value.to_le_bytes());
		}
		Ok(PreparedSecretKeygen {
			device,
			pipeline,
			_seed: seed,
			_private: private,
			public,
			sets,
			push,
		})
	}
}

impl PreparedSecretKeygen<'_> {
	pub(super) fn encode(&self, command: ash::vk::CommandBuffer) {
		let set = self
			.sets
			.as_ref()
			.map_or_else(|| self.device.descriptor_set(), |sets| sets.set(0));
		// SAFETY: the Engine owns this recording command, admitted pipeline and
		// descriptor layout. Preflight proved ownership, disjoint exact ranges,
		// the 16-byte push ABI and [1,1,1] dispatch limits. Borrowed secrets and
		// retained public storage keep every descriptor live. The upload recorder
		// supplies COPY/TRANSFER_WRITE -> compute visibility for the seed.
		unsafe {
			self.device.raw().cmd_bind_pipeline(
				command,
				ash::vk::PipelineBindPoint::COMPUTE,
				self.pipeline.raw(),
			);
			self.device.raw().cmd_bind_descriptor_sets(
				command,
				ash::vk::PipelineBindPoint::COMPUTE,
				self.pipeline.layout(),
				0,
				&[set],
				&[],
			);
			self.device.raw().cmd_push_constants(
				command,
				self.pipeline.layout(),
				ash::vk::ShaderStageFlags::COMPUTE,
				0,
				&self.push,
			);
			self.device.raw().cmd_dispatch(command, 1, 1, 1);
		}
		let barrier = ash::vk::BufferMemoryBarrier2::default()
			.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
			.src_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_WRITE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
			.dst_access_mask(ash::vk::AccessFlags2::HOST_READ)
			.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
			.buffer(self.public.raw())
			.size(self.public.size());
		// SAFETY: public encoding is the only host-readable result. Exact Engine
		// completion and Buffer::read invalidation precede host observation.
		unsafe {
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&ash::vk::DependencyInfo::default().buffer_memory_barriers(&[barrier]),
			);
		}
	}

	pub(super) fn retention(self) -> (Buffer, Option<BoundedSets>) {
		(self.public, self.sets)
	}
}

/// Private PQC consumer operands. Secret classification is fixed by the constructors,
/// rather than supplied by an ordinary dispatch or a caller-owned binding list.
pub(super) struct SecretConsumerOperands<'a> {
	device: &'a Arc<Device>,
	kernel: KernelId,
	_secrets: [SecretBinding<'a>; 2],
	_workspace: Option<SecretBinding<'a>>,
	public: Vec<Buffer>,
	bindings: Vec<ash::vk::DescriptorBufferInfo>,
	indices: Vec<u32>,
	host_output_start: usize,
	scalars: Vec<u32>,
}

pub(in crate::runtime) struct PreparedSecretConsumer<'a> {
	operands: SecretConsumerOperands<'a>,
	pipeline: &'a ComputePipeline,
	sets: Option<BoundedSets>,
	push: Vec<u8>,
}

impl<'a> SecretConsumerOperands<'a> {
	pub(super) fn mldsa_sign(
		device: &'a Arc<Device>,
		secrets: [SecretBinding<'a>; 3],
		public: [&Buffer; 3],
		parameter: u32,
	) -> Result<Self> {
		let [seed, private, workspace] = secrets;
		let (seed_len, private_len, mu_len, sig_len) = KernelId::mldsa_sign_layout(parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		Self::validate_secret(device, &seed, seed_len)?;
		Self::validate_secret(device, &private, private_len)?;
		Self::validate_secret(
			device,
			&workspace,
			KernelId::mldsa_sign_workspace_bytes() as u64,
		)?;
		let [mu, signature, status] = public;
		Self::validate_public(device, mu, mu_len)?;
		Self::validate_public(device, signature, (sig_len + 3) & !3)?;
		Self::validate_public(device, status, 4)?;
		let bindings = vec![
			Self::secret_info(&seed),
			Self::secret_info(&private),
			Self::public_info(mu),
			Self::public_info(signature),
			Self::public_info(status),
			Self::secret_info(&workspace),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			seed.descriptor_index(),
			private.descriptor_index(),
			mu.descriptor_index(),
			signature.descriptor_index(),
			status.descriptor_index(),
			workspace.descriptor_index(),
		];
		Ok(Self {
			device,
			kernel: KernelId::CryptographyMlDsaSignU8,
			_secrets: [seed, private],
			_workspace: Some(workspace),
			public: public.into_iter().cloned().collect(),
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![parameter],
		})
	}

	pub(super) fn mldsa_sign_message(
		device: &'a Arc<Device>,
		secrets: [SecretBinding<'a>; 3],
		public: [&Buffer; 4],
		parameter: u32,
		lengths: [u32; 2],
		algorithm: Option<u32>,
	) -> Result<Self> {
		if let Some(algorithm) = algorithm {
			KernelId::mldsa_prehash_bytes(algorithm)
				.ok_or_else(|| Error::invalid_argument("ML-DSA prehash identity is unknown"))?;
		}
		let [seed, private, workspace] = secrets;
		let (seed_len, private_len, _, sig_len) = KernelId::mldsa_sign_layout(parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		Self::validate_secret(device, &seed, seed_len)?;
		Self::validate_secret(device, &private, private_len)?;
		Self::validate_secret(
			device,
			&workspace,
			KernelId::mldsa_sign_workspace_bytes() as u64,
		)?;
		let [message, context, signature, status] = public;
		if lengths[1] > 255 {
			return Err(Error::invalid_argument("ML-DSA context exceeds 255 bytes"));
		}
		for (input, length) in [(message, lengths[0]), (context, lengths[1])] {
			let extent = length
				.checked_add(3)
				.ok_or_else(|| Error::invalid_argument("ML-DSA input extent overflow"))?;
			Self::validate_public(device, input, u64::from((extent & !3).max(4)))?;
		}
		Self::validate_public(device, signature, (sig_len + 3) & !3)?;
		Self::validate_public(device, status, 4)?;
		let bindings = vec![
			Self::secret_info(&seed),
			Self::secret_info(&private),
			Self::public_info(message),
			Self::public_info(context),
			Self::public_info(signature),
			Self::public_info(status),
			Self::secret_info(&workspace),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			seed.descriptor_index(),
			private.descriptor_index(),
			message.descriptor_index(),
			context.descriptor_index(),
			signature.descriptor_index(),
			status.descriptor_index(),
			workspace.descriptor_index(),
		];
		let mut scalars = vec![parameter, lengths[0], lengths[1]];
		if let Some(algorithm) = algorithm {
			scalars.push(algorithm);
		}
		Ok(Self {
			device,
			kernel: if algorithm.is_some() {
				KernelId::CryptographyMlDsaSignHashMessageU8
			} else {
				KernelId::CryptographyMlDsaSignMessageU8
			},
			_secrets: [seed, private],
			_workspace: Some(workspace),
			public: public.into_iter().cloned().collect(),
			bindings,
			indices,
			host_output_start: 2,
			scalars,
		})
	}
	pub(super) fn mldsa_sign_prehashed(
		device: &'a Arc<Device>,
		secrets: [SecretBinding<'a>; 3],
		public: [&Buffer; 4],
		parameter: u32,
		algorithm: u32,
		context_length: u32,
	) -> Result<Self> {
		let [seed, private, workspace] = secrets;
		let (seed_len, private_len, _, sig_len) = KernelId::mldsa_sign_layout(parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		Self::validate_secret(device, &seed, seed_len)?;
		Self::validate_secret(device, &private, private_len)?;
		Self::validate_secret(
			device,
			&workspace,
			KernelId::mldsa_sign_workspace_bytes() as u64,
		)?;
		let [digest, context, signature, status] = public;
		let digest_len = KernelId::mldsa_prehash_bytes(algorithm)
			.ok_or_else(|| Error::invalid_argument("ML-DSA prehash identity is unknown"))?;
		if context_length > 255 {
			return Err(Error::invalid_argument("ML-DSA context exceeds 255 bytes"));
		}
		Self::validate_public(device, digest, digest_len)?;
		Self::validate_public(
			device,
			context,
			u64::from(((context_length + 3) & !3).max(4)),
		)?;
		Self::validate_public(device, signature, (sig_len + 3) & !3)?;
		Self::validate_public(device, status, 4)?;
		let bindings = vec![
			Self::secret_info(&seed),
			Self::secret_info(&private),
			Self::public_info(digest),
			Self::public_info(context),
			Self::public_info(signature),
			Self::public_info(status),
			Self::secret_info(&workspace),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			seed.descriptor_index(),
			private.descriptor_index(),
			digest.descriptor_index(),
			context.descriptor_index(),
			signature.descriptor_index(),
			status.descriptor_index(),
			workspace.descriptor_index(),
		];
		Ok(Self {
			device,
			kernel: KernelId::CryptographyMlDsaSignPrehashedU8,
			_secrets: [seed, private],
			_workspace: Some(workspace),
			public: public.into_iter().cloned().collect(),
			bindings,
			indices,
			host_output_start: 2,
			scalars: vec![parameter, algorithm, context_length],
		})
	}

	pub(super) fn encaps(
		device: &'a Arc<Device>,
		seed: SecretBinding<'a>,
		public: &Buffer,
		shared: SecretBinding<'a>,
		ciphertext: &Buffer,
		status: &Buffer,
		k: u32,
	) -> Result<Self> {
		Self::validate_parameter(k)?;
		Self::validate_secret(device, &seed, 32)?;
		Self::validate_secret(device, &shared, 32)?;
		Self::validate_public(device, public, u64::from(384 * k + 32))?;
		Self::validate_public(device, ciphertext, Self::ciphertext_size(k))?;
		Self::validate_public(device, status, 4)?;
		let bindings = vec![
			Self::secret_info(&seed),
			Self::public_info(public),
			Self::secret_info(&shared),
			Self::public_info(ciphertext),
			Self::public_info(status),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			seed.descriptor_index(),
			public.descriptor_index(),
			shared.descriptor_index(),
			ciphertext.descriptor_index(),
			status.descriptor_index(),
		];
		Ok(Self {
			device,
			kernel: KernelId::CryptographyMlKemEncapsU8,
			_secrets: [seed, shared],
			_workspace: None,
			public: vec![public.clone(), ciphertext.clone(), status.clone()],
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![k],
		})
	}

	pub(super) fn decaps(
		device: &'a Arc<Device>,
		private: SecretBinding<'a>,
		ciphertext: &Buffer,
		shared: SecretBinding<'a>,
		k: u32,
	) -> Result<Self> {
		Self::validate_parameter(k)?;
		Self::validate_secret(device, &private, u64::from(768 * k + 96))?;
		Self::validate_secret(device, &shared, 32)?;
		Self::validate_public(device, ciphertext, Self::ciphertext_size(k))?;
		let bindings = vec![
			Self::secret_info(&private),
			Self::public_info(ciphertext),
			Self::secret_info(&shared),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			private.descriptor_index(),
			ciphertext.descriptor_index(),
			shared.descriptor_index(),
		];
		Ok(Self {
			device,
			kernel: KernelId::CryptographyMlKemDecapsU8,
			_secrets: [private, shared],
			_workspace: None,
			public: vec![ciphertext.clone()],
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![k],
		})
	}

	fn validate_parameter(k: u32) -> Result<()> {
		if !(2..=4).contains(&k) {
			return Err(Error::invalid_argument("ML-KEM k must be 2, 3 or 4"));
		}
		Ok(())
	}

	fn ciphertext_size(k: u32) -> u64 {
		u64::from(32 * if k == 4 { 11 * k + 5 } else { 10 * k + 4 })
	}

	fn validate_secret(device: &Arc<Device>, secret: &SecretBinding<'_>, size: u64) -> Result<()> {
		if !secret.belongs_to(device) || secret.byte_len() != size {
			return Err(Error::invalid_argument(
				"PQC secret ownership or extent mismatch",
			));
		}
		Ok(())
	}

	fn validate_public(device: &Arc<Device>, public: &Buffer, size: u64) -> Result<()> {
		if !public.belongs_to(device) || public.size() != size {
			return Err(Error::invalid_argument(
				"PQC public ownership or extent mismatch",
			));
		}
		Ok(())
	}

	fn secret_info(secret: &SecretBinding<'_>) -> ash::vk::DescriptorBufferInfo {
		ash::vk::DescriptorBufferInfo::default()
			.buffer(secret.raw())
			.range(secret.byte_len())
	}

	fn public_info(public: &Buffer) -> ash::vk::DescriptorBufferInfo {
		ash::vk::DescriptorBufferInfo::default()
			.buffer(public.raw())
			.range(public.size())
	}

	fn validate_disjoint(bindings: &[ash::vk::DescriptorBufferInfo]) -> Result<()> {
		for (index, binding) in bindings.iter().enumerate() {
			if bindings[..index]
				.iter()
				.any(|other| other.buffer == binding.buffer)
			{
				return Err(Error::invalid_argument("PQC operands must be disjoint"));
			}
		}
		Ok(())
	}

	pub(super) fn belongs_to(&self, device: &Arc<Device>) -> bool {
		self.device.same_as(device)
	}

	pub(super) fn kernel(&self) -> KernelId {
		self.kernel
	}

	pub(super) fn prepare(self, pipeline: &'a ComputePipeline) -> Result<PreparedSecretConsumer<'a>> {
		let size = u32::try_from((self.indices.len() + self.scalars.len()) * 4)
			.map_err(|_| Error::invalid_argument("PQC push extent overflow"))?;
		if pipeline.push_constant_size() != size || pipeline.max_dispatch_group_count().contains(&0) {
			return Err(Error::missing_capability("PQC dispatch ABI unavailable"));
		}
		let bounded = self.device.physical().profile == ExecutionProfile::Compatibility;
		let sets = if bounded {
			let layout = self
				.device
				.bounded_descriptor_layout(self.kernel.bounded_buffer_count())
				.ok_or_else(|| Error::missing_capability("PQC bounded descriptor layout unavailable"))?;
			Some(BoundedSets::new(
				self.device,
				&[layout],
				std::slice::from_ref(&self.bindings),
			)?)
		} else {
			None
		};
		let mut push = Vec::with_capacity(
			usize::try_from(size).map_err(|_| Error::invalid_argument("PQC push extent overflow"))?,
		);
		for (slot, index) in self.indices.iter().enumerate() {
			let value = if bounded {
				u32::try_from(slot).map_err(|_| Error::invalid_argument("PQC binding index overflow"))?
			} else {
				*index
			};
			push.extend_from_slice(&value.to_le_bytes());
		}
		for scalar in &self.scalars {
			push.extend_from_slice(&scalar.to_le_bytes());
		}
		Ok(PreparedSecretConsumer {
			operands: self,
			pipeline,
			sets,
			push,
		})
	}
}

impl PreparedSecretConsumer<'_> {
	pub(super) fn encode(&self, command: ash::vk::CommandBuffer) {
		let device = self.operands.device;
		let set = self
			.sets
			.as_ref()
			.map_or_else(|| device.descriptor_set(), |sets| sets.set(0));
		// A queue order is not a memory dependency. Include prior key generation,
		// ciphertext production and host uploads before this one-shot consumer.
		let dependency = ash::vk::MemoryBarrier2::default()
			.src_stage_mask(
				ash::vk::PipelineStageFlags2::ALL_COMMANDS | ash::vk::PipelineStageFlags2::HOST,
			)
			.src_access_mask(ash::vk::AccessFlags2::MEMORY_WRITE | ash::vk::AccessFlags2::HOST_WRITE)
			.dst_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
			.dst_access_mask(
				ash::vk::AccessFlags2::SHADER_STORAGE_READ | ash::vk::AccessFlags2::SHADER_STORAGE_WRITE,
			);
		// SAFETY: checked constructors prove disjoint exact ranges and originating
		// Device identity. The one-shot recorder retains every resource and set
		// until exact completion; borrowed secret bindings remain live here.
		unsafe {
			device.sync_commands().pipeline_barrier(
				device.raw(),
				command,
				&ash::vk::DependencyInfo::default().memory_barriers(&[dependency]),
			);
			device.raw().cmd_bind_pipeline(
				command,
				ash::vk::PipelineBindPoint::COMPUTE,
				self.pipeline.raw(),
			);
			device.raw().cmd_bind_descriptor_sets(
				command,
				ash::vk::PipelineBindPoint::COMPUTE,
				self.pipeline.layout(),
				0,
				&[set],
				&[],
			);
			device.raw().cmd_push_constants(
				command,
				self.pipeline.layout(),
				ash::vk::ShaderStageFlags::COMPUTE,
				0,
				&self.push,
			);
			device.raw().cmd_dispatch(command, 1, 1, 1);
		}
		let barriers: Vec<_> = self.operands.public[self.operands.host_output_start..]
			.iter()
			.map(|buffer| {
				ash::vk::BufferMemoryBarrier2::default()
					.src_stage_mask(ash::vk::PipelineStageFlags2::COMPUTE_SHADER)
					.src_access_mask(ash::vk::AccessFlags2::SHADER_STORAGE_WRITE)
					.dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
					.dst_access_mask(ash::vk::AccessFlags2::HOST_READ)
					.src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
					.dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
					.buffer(buffer.raw())
					.size(buffer.size())
			})
			.collect();
		if !barriers.is_empty() {
			// SAFETY: only explicitly classified public outputs become host-visible.
			unsafe {
				device.sync_commands().pipeline_barrier(
					device.raw(),
					command,
					&ash::vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
				);
			}
		}
	}

	pub(super) fn retention(self) -> (Vec<Buffer>, Option<BoundedSets>) {
		(self.operands.public, self.sets)
	}

	pub(super) fn retain(self, command: &mut crate::runtime::RecordedCommandBuffer) -> Result<()> {
		let (public, sets) = self.retention();
		command.retain_secret_public_resources(public, sets)
	}
}
