//! Private PQC recording through the Engine-owned pipeline registry.
//! FIPS 203/204 formulas are imported unchanged; C++ oaPqc.md has no live donor.

use std::sync::Arc;

use super::{descriptor::BoundedSets, pipeline::ComputePipeline, secret_buffer::SecretBinding};
use crate::runtime::{Buffer, Device};
use crate::{Error, Result, runtime::dna::kernel::KernelId};

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
	push: Vec<u8>,
}

/// Ownership and wire admission precede fallible pipeline creation.
pub(super) struct SecretKeygenOperands<'a> {
	device: &'a Arc<Device>,
	seed: SecretBinding<'a>,
	private: SecretBinding<'a>,
	public: Buffer,
	parameter: u32,
	kernel: KernelId,
	#[cfg(oa_profile = "compatibility")]
	seed_len: u64,
	#[cfg(oa_profile = "compatibility")]
	el: u64,
	#[cfg(oa_profile = "compatibility")]
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
			#[cfg(oa_profile = "compatibility")]
			seed_len,
			#[cfg(oa_profile = "compatibility")]
			el,
			#[cfg(oa_profile = "compatibility")]
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
			#[cfg(oa_profile = "compatibility")]
			kernel,
			#[cfg(oa_profile = "core")]
				kernel: _,
			#[cfg(oa_profile = "compatibility")]
			seed_len,
			#[cfg(oa_profile = "compatibility")]
			el,
			#[cfg(oa_profile = "compatibility")]
			dl,
		} = self;
		if pipeline.push_constant_size() != 16 || pipeline.max_dispatch_group_count().contains(&0) {
			return Err(Error::missing_capability("PQC dispatch ABI unavailable"));
		}
		let bounded = cfg!(oa_profile = "compatibility");
		#[cfg(oa_profile = "compatibility")]
		let sets = {
			let layout = device
				.bounded_descriptor_layout(kernel.bounded_buffer_count())
				.ok_or_else(|| Error::missing_capability("PQC bounded descriptor layout unavailable"))?;
			let buffers = vec![
				vk::DescriptorBufferInfo::new()
					.set_buffer(seed.raw())
					.set_range(seed_len),
				vk::DescriptorBufferInfo::new()
					.set_buffer(private.raw())
					.set_range(dl),
				vk::DescriptorBufferInfo::new()
					.set_buffer(public.raw())
					.set_range(el),
			];
			// BoundedSets retains the Device and descriptors until command retirement.
			Some(BoundedSets::new(
				device,
				&[layout],
				std::slice::from_ref(&buffers),
			)?)
		};
		#[cfg(oa_profile = "core")]
		let sets = None;
		let indices = if bounded {
			vec![0, 1, 2]
		} else {
			vec![
				seed.descriptor_index(),
				private.descriptor_index(),
				public.descriptor_index(),
			]
		};
		let mut push = vec![0; 16];
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
	pub(super) fn encode(&self, command: vk::CommandBuffer) -> Result<()> {
		#[cfg(oa_profile = "compatibility")]
		let set = self
			.sets
			.as_ref()
			.map_or_else(|| self.device.descriptor_set(), |sets| sets.set(0));
		#[cfg(oa_profile = "core")]
		let set = self.device.descriptor_set();
		// SAFETY: the Engine owns this recording command, admitted pipeline and
		// descriptor layout. Preflight proved ownership, disjoint exact ranges,
		// the parameter-family push ABI and [1,1,1] dispatch limits. Borrowed secrets and
		// retained public storage keep every descriptor live. The upload recorder
		// supplies COPY/TRANSFER_WRITE -> compute visibility for the seed.
		unsafe {
			self.device.raw().cmd_bind_pipeline(
				command,
				vk::PipelineBindPoint::COMPUTE,
				self.pipeline.raw(),
			)?;
			self.device.raw().cmd_bind_descriptor_sets(
				command,
				vk::PipelineBindPoint::COMPUTE,
				self.pipeline.layout(),
				0,
				&[set],
				&[],
			)?;
			self.device.raw().cmd_push_constants(
				command,
				self.pipeline.layout(),
				vk::ShaderStageFlagBits::COMPUTE_BIT,
				0,
				&self.push,
			)?;
			self.device.raw().cmd_dispatch(command, 1, 1, 1)?;
		}
		let barrier = vk::BufferMemoryBarrier2::new()
			.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
			.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
			.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT)
			.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.set_buffer(self.public.raw())
			.set_size(self.public.size());
		// SAFETY: public encoding is the only host-readable result. Exact Engine
		// completion and Buffer::read invalidation precede host observation.
		unsafe {
			self.device.sync_commands().pipeline_barrier(
				self.device.raw(),
				command,
				&(vk::DependencyInfo::new()
					.set_buffer_memory_barriers(&[barrier])
					.map_err(Error::from)?),
			)?;
		}
		Ok(())
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
	_secrets: Vec<SecretBinding<'a>>,
	_workspace: Option<SecretBinding<'a>>,
	public: Vec<Buffer>,
	#[cfg(oa_profile = "compatibility")]
	bindings: Vec<vk::DescriptorBufferInfo>,
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
			_secrets: vec![seed, private],
			_workspace: Some(workspace),
			public: public.into_iter().cloned().collect(),
			#[cfg(oa_profile = "compatibility")]
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![parameter],
		})
	}

	/// Public mu = H(tr || M', 64) derivation operands. Every binding is public
	/// input or internal scratch; the private signer consumes mu separately.
	pub(super) fn mldsa_mu(
		device: &'a Arc<Device>,
		public: [&Buffer; 4],
		parameter: u32,
		lengths: [u32; 2],
		algorithm: u32,
	) -> Result<Self> {
		let (_, public_len, _) = KernelId::mldsa_keygen_layout(parameter)
			.ok_or_else(|| Error::invalid_argument("ML-DSA parameter set must be 44, 65 or 87"))?;
		if lengths[1] > 255 {
			return Err(Error::invalid_argument("ML-DSA context exceeds 255 bytes"));
		}
		let (head, head_length) = KernelId::mldsa_mu_head(algorithm, lengths[1])
			.ok_or_else(|| Error::invalid_argument("ML-DSA prehash identity is unknown"))?;
		let payload_extent = if algorithm == 0 {
			let extent = lengths[0]
				.checked_add(3)
				.ok_or_else(|| Error::invalid_argument("ML-DSA input extent overflow"))?;
			u64::from((extent & !3).max(4))
		} else {
			let length = KernelId::mldsa_prehash_bytes(algorithm)
				.ok_or_else(|| Error::invalid_argument("ML-DSA prehash identity is unknown"))?;
			if lengths[0] != u32::try_from(length).unwrap_or(u32::MAX) {
				return Err(Error::invalid_argument("ML-DSA digest extent mismatch"));
			}
			length
		};
		let [public_key, payload, context, mu] = public;
		Self::validate_public(device, public_key, public_len)?;
		Self::validate_public(device, payload, payload_extent)?;
		Self::validate_public(device, context, u64::from(((lengths[1] + 3) & !3).max(4)))?;
		Self::validate_public(device, mu, 64)?;
		let bindings = vec![
			Self::public_info(public_key),
			Self::public_info(payload),
			Self::public_info(context),
			Self::public_info(mu),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			public_key.descriptor_index(),
			payload.descriptor_index(),
			context.descriptor_index(),
			mu.descriptor_index(),
		];
		Ok(Self {
			device,
			kernel: KernelId::CryptographyMlDsaMuU8,
			_secrets: Vec::new(),
			_workspace: None,
			public: public.into_iter().cloned().collect(),
			#[cfg(oa_profile = "compatibility")]
			bindings,
			indices,
			host_output_start: 4,
			scalars: vec![
				parameter,
				lengths[0],
				lengths[1],
				head_length,
				head[0],
				head[1],
				head[2],
				head[3],
			],
		})
	}

	pub(super) fn encaps(
		device: &'a Arc<Device>,
		secrets: [SecretBinding<'a>; 2],
		public: &Buffer,
		ciphertext: &Buffer,
		status: &Buffer,
		k: u32,
	) -> Result<Self> {
		let [seed, shared] = secrets;
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
			_secrets: vec![seed, shared],
			_workspace: None,
			public: vec![public.clone(), ciphertext.clone(), status.clone()],
			#[cfg(oa_profile = "compatibility")]
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![k],
		})
	}

	pub(super) fn decaps(
		device: &'a Arc<Device>,
		secrets: [SecretBinding<'a>; 2],
		ciphertext: &Buffer,
		k: u32,
	) -> Result<Self> {
		let [private, shared] = secrets;
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
			_secrets: vec![private, shared],
			_workspace: None,
			public: vec![ciphertext.clone()],
			#[cfg(oa_profile = "compatibility")]
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![k],
		})
	}

	/// One-shot declassification consumer: the 32-byte shared secret stays in
	/// private storage while a keyed hash emits only public derived bytes.
	pub(super) fn shared_derive(
		device: &'a Arc<Device>,
		shared: SecretBinding<'a>,
		info: &Buffer,
		output: &Buffer,
		info_len: u32,
		output_len: u32,
	) -> Result<Self> {
		Self::validate_secret(device, &shared, 32)?;
		if output_len == 0 || output_len > 4096 || !output_len.is_multiple_of(4) {
			return Err(Error::invalid_argument(
				"shared-secret derivation extent must be a nonzero multiple of four within 4096",
			));
		}
		Self::validate_public(device, output, u64::from(output_len))?;
		let info_extent = u64::from(((info_len + 3) & !3).max(4));
		if !info.belongs_to(device) || info.size() < info_extent {
			return Err(Error::invalid_argument(
				"shared-secret info ownership or extent mismatch",
			));
		}
		let bindings = vec![
			Self::secret_info(&shared),
			Self::public_info(info),
			Self::public_info(output),
		];
		Self::validate_disjoint(&bindings)?;
		let indices = vec![
			shared.descriptor_index(),
			info.descriptor_index(),
			output.descriptor_index(),
		];
		Ok(Self {
			device,
			kernel: KernelId::CryptographyMlKemSharedDeriveU8,
			_secrets: vec![shared],
			_workspace: None,
			public: vec![info.clone(), output.clone()],
			#[cfg(oa_profile = "compatibility")]
			bindings,
			indices,
			host_output_start: 1,
			scalars: vec![info_len, output_len],
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

	fn secret_info(secret: &SecretBinding<'_>) -> vk::DescriptorBufferInfo {
		vk::DescriptorBufferInfo::new()
			.set_buffer(secret.raw())
			.set_range(secret.byte_len())
	}

	fn public_info(public: &Buffer) -> vk::DescriptorBufferInfo {
		vk::DescriptorBufferInfo::new()
			.set_buffer(public.raw())
			.set_range(public.size())
	}

	fn validate_disjoint(bindings: &[vk::DescriptorBufferInfo]) -> Result<()> {
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
		let bounded = cfg!(oa_profile = "compatibility");
		#[cfg(oa_profile = "compatibility")]
		let sets = {
			let layout = self
				.device
				.bounded_descriptor_layout(self.kernel.bounded_buffer_count())
				.ok_or_else(|| Error::missing_capability("PQC bounded descriptor layout unavailable"))?;
			Some(BoundedSets::new(
				self.device,
				&[layout],
				std::slice::from_ref(&self.bindings),
			)?)
		};
		#[cfg(oa_profile = "core")]
		let sets = None;
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
	pub(super) fn encode(&self, command: vk::CommandBuffer) -> Result<()> {
		let device = self.operands.device;
		#[cfg(oa_profile = "compatibility")]
		let set = self
			.sets
			.as_ref()
			.map_or_else(|| device.descriptor_set(), |sets| sets.set(0));
		#[cfg(oa_profile = "core")]
		let set = device.descriptor_set();
		// A queue order is not a memory dependency. Include prior key generation,
		// ciphertext production and host uploads before this one-shot consumer.
		let dependency = vk::MemoryBarrier2::new()
			.set_src_stage_mask(
				vk::PipelineStageFlagBits2::ALL_COMMANDS_BIT | vk::PipelineStageFlagBits2::HOST_BIT,
			)
			.set_src_access_mask(
				vk::AccessFlagBits2::MEMORY_WRITE_BIT | vk::AccessFlagBits2::HOST_WRITE_BIT,
			)
			.set_dst_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
			.set_dst_access_mask(
				vk::AccessFlagBits2::SHADER_STORAGE_READ_BIT
					| vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT,
			);
		// SAFETY: checked constructors prove disjoint exact ranges and originating
		// Device identity. The one-shot recorder retains every resource and set
		// until exact completion; borrowed secret bindings remain live here.
		unsafe {
			device.sync_commands().pipeline_barrier(
				device.raw(),
				command,
				&(vk::DependencyInfo::new()
					.set_memory_barriers(&[dependency])
					.map_err(Error::from)?),
			)?;
			device.raw().cmd_bind_pipeline(
				command,
				vk::PipelineBindPoint::COMPUTE,
				self.pipeline.raw(),
			)?;
			device.raw().cmd_bind_descriptor_sets(
				command,
				vk::PipelineBindPoint::COMPUTE,
				self.pipeline.layout(),
				0,
				&[set],
				&[],
			)?;
			device.raw().cmd_push_constants(
				command,
				self.pipeline.layout(),
				vk::ShaderStageFlagBits::COMPUTE_BIT,
				0,
				&self.push,
			)?;
			device.raw().cmd_dispatch(command, 1, 1, 1)?;
		}
		let barriers: Vec<_> = self.operands.public[self.operands.host_output_start..]
			.iter()
			.map(|buffer| {
				vk::BufferMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
					.set_dst_stage_mask(vk::PipelineStageFlagBits2::HOST_BIT)
					.set_dst_access_mask(vk::AccessFlagBits2::HOST_READ_BIT)
					.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.set_buffer(buffer.raw())
					.set_size(buffer.size())
			})
			.collect();
		if !barriers.is_empty() {
			// SAFETY: only explicitly classified public outputs become host-visible.
			unsafe {
				device.sync_commands().pipeline_barrier(
					device.raw(),
					command,
					&(vk::DependencyInfo::new()
						.set_buffer_memory_barriers(&barriers)
						.map_err(Error::from)?),
				)?;
			}
		}
		Ok(())
	}

	pub(super) fn retention(self) -> (Vec<Buffer>, Option<BoundedSets>) {
		(self.operands.public, self.sets)
	}

	pub(super) fn retain(self, command: &mut crate::runtime::RecordedCommandBuffer) -> Result<()> {
		let (public, sets) = self.retention();
		command.retain_secret_public_resources(public, sets)
	}
}

#[cfg(test)]
#[path = "../../../../../test/rs/runtime/pqc_unit.rs"]
pub(in crate::runtime) mod tests;
