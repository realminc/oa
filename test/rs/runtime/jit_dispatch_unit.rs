//! Validated variable-ABI laboratory recorder, sharing the Engine's pool, descriptors and cache.

use super::*;
use crate::runtime::dna::{elementwise::Program, kernel::jit::CompiledChain};

/// Per-program dispatch extents kept outside the shared code artifact.
/// A zero input extent means a linear full-count read, as in the manual
/// Phase 1 recipe artifact which carries no IR identity.
pub(in crate::runtime) struct ChainContract {
	input_bytes: Vec<u64>,
	output_count: usize,
	artifact_key: Vec<u8>,
}

impl ChainContract {
	pub(in crate::runtime) fn fixed(input_count: usize, output_count: usize) -> Self {
		Self {
			input_bytes: vec![0; input_count],
			output_count,
			artifact_key: Vec::new(),
		}
	}

	pub(in crate::runtime) fn of(program: &Program) -> Self {
		Self {
			input_bytes: program.input_bytes.clone(),
			output_count: program.outputs.len(),
			artifact_key: program.artifact_key(),
		}
	}
}

pub(in crate::runtime) struct PreparedChain {
	pipeline: Arc<ComputePipeline>,
	buffers: Vec<Buffer>,
	input_count: usize,
	count: u32,
	device: Arc<Device>,
}

impl PreparedChain {
	pub(in crate::runtime) fn new(
		device: &Arc<Device>,
		artifact: &CompiledChain,
		contract: &ChainContract,
		buffers: impl Into<Vec<Buffer>>,
		count: u32,
	) -> Result<Self> {
		let buffers = buffers.into();
		let input_count = contract.input_bytes.len();
		let bounded = device.physical().profile == ExecutionProfile::Compatibility;
		if artifact.bounded != bounded || artifact.key != contract.artifact_key {
			return Err(Error::invalid_argument(
				"specialization descriptor profile or artifact identity mismatch",
			));
		}
		let bytes = u64::from(count) * 4;
		if bytes > u64::from(u32::MAX)
			|| buffers.len() != input_count + contract.output_count
			|| buffers.len() > 31
			|| buffers.iter().any(|buffer| !buffer.belongs_to(device))
			|| buffers
				.iter()
				.take(input_count)
				.zip(&contract.input_bytes)
				.any(|(buffer, &minimum)| buffer.size() < if minimum == 0 { bytes } else { minimum })
			|| buffers
				.iter()
				.skip(input_count)
				.any(|buffer| buffer.size() < bytes)
			|| buffers
				.iter()
				.enumerate()
				.skip(input_count)
				.any(|(i, output)| buffers[..i].iter().any(|input| input.same_as(output)))
		{
			return Err(Error::invalid_argument(
				"specialization buffer extent, owner or alias mismatch",
			));
		}
		if count.div_ceil(256).max(1) > device.max_compute_work_group_count()[0] {
			return Err(Error::invalid_argument(
				"specialization exceeds one-dimensional dispatch limit",
			));
		}
		let layout = if bounded {
			device
				.bounded_descriptor_layout(buffers.len() as u32)
				.ok_or_else(|| Error::missing_capability("specialization bounded layout unavailable"))?
		} else {
			device.descriptor_layout()
		};
		let cache = device.pipeline_cache()?;
		let pipeline = ComputePipeline::from_words(
			device.raw_arc(),
			layout,
			artifact.words(),
			[256, 1, 1],
			device.physical().limits,
			cache.raw(),
		)?;
		Ok(Self {
			pipeline: Arc::new(pipeline),
			buffers,
			input_count,
			count,
			device: device.clone(),
		})
	}

	pub(in crate::runtime::device) fn record(
		&self,
		pool: &mut CommandPool,
		device: &Arc<Device>,
	) -> Result<ReusableCommandBuffer> {
		if !device.same_as(&self.device) {
			return Err(Error::invalid_argument(
				"specialization recorder Engine mismatch",
			));
		}
		let bounded = device.physical().profile == ExecutionProfile::Compatibility;
		let sets = if bounded {
			let layout = device
				.bounded_descriptor_layout(self.buffers.len() as u32)
				.expect("admitted layout");
			let infos = self
				.buffers
				.iter()
				.map(|buffer| {
					vk::DescriptorBufferInfo::new()
						.set_buffer(buffer.raw())
						.set_range(buffer.size())
				})
				.collect();
			Some(BoundedSets::new(device, &[layout], &[infos])?)
		} else {
			None
		};
		let set = sets
			.as_ref()
			.map_or_else(|| device.descriptor_set(), |sets| sets.set(0));
		let indices: Vec<u32> = if bounded {
			(0..self.buffers.len() as u32).collect()
		} else {
			self.buffers.iter().map(Buffer::descriptor_index).collect()
		};
		let push = encode_push_constants(&[PushConstant::U32(self.count)], &indices, "jit_chain_lab")?;
		if push.len() != self.pipeline.push_constant_size() as usize {
			return Err(Error::invalid_argument("specialization push ABI mismatch"));
		}
		let command = pool.begin(
			device.raw(),
			vk::CommandBufferUsageFlagBits::SIMULTANEOUS_USE_BIT,
		)?;
		let encode = || -> Result<()> {
			// SAFETY: exact ABI, device ownership, ranges and one-dimensional limits
			// were checked. Invocation i owns only output[i], tails exit before I/O.
			// All resources and the pipeline are retained by the recorded command.
			unsafe {
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
					&push,
				)?;
				device
					.raw()
					.cmd_dispatch(command, self.count.div_ceil(256).max(1), 1, 1)?;
			}
			record_host_visibility(
				device.raw(),
				device.sync_commands(),
				command,
				&[BufferAccess::Write],
			)
		};
		if let Err(error) = encode() {
			pool.free_handle(device.raw(), command);
			return Err(error);
		}
		let recorded = pool.finish(
			device.raw(),
			command,
			CommandRetention {
				resources: self.buffers.to_vec(),
				accesses: (0..self.buffers.len())
					.map(|i| {
						if i < self.input_count {
							BufferAccess::Read
						} else {
							BufferAccess::Write
						}
					})
					.collect(),
				bounded_sets: sets,
				pipelines: vec![self.pipeline.clone()],
				route_timings: None,
			},
			None,
			Some(device.clone()),
		)?;
		let CommandBufferAllocation::Reusable(reusable) = recorded.allocation else {
			unreachable!()
		};
		Ok(reusable)
	}
}
