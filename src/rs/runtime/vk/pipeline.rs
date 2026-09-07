use crate::{Error, Result, runtime::shader::ShaderArtifact};

use super::physical::DeviceLimits;

pub(super) struct ComputePipeline {
	handle: ash::vk::Pipeline,
	layout: ash::vk::PipelineLayout,
	max_dispatch_group_count: [u32; 3],
	push_constant_size: u32,
}

impl ComputePipeline {
	pub(super) fn new(
		device: &ash::Device,
		descriptor_layout: ash::vk::DescriptorSetLayout,
		artifact: &'static ShaderArtifact,
		limits: DeviceLimits,
	) -> Result<Self> {
		validate_artifact_limits(artifact.workgroup_size, artifact.push_constant_size, limits)?;

		let descriptor_layouts = [descriptor_layout];
		let push_range = ash::vk::PushConstantRange::default()
			.stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
			.offset(0)
			.size(artifact.push_constant_size);
		let push_ranges = [push_range];
		let layout_info = ash::vk::PipelineLayoutCreateInfo::default()
			.set_layouts(&descriptor_layouts)
			.push_constant_ranges(&push_ranges);
		// SAFETY: the shared descriptor layout is live and the reflected push range
		// fits the queried device limit and Vulkan's four-byte alignment requirements.
		let layout =
			unsafe { device.create_pipeline_layout(&layout_info, None) }.map_err(|source| {
				Error::backend_failure("Vulkan", "compute pipeline-layout creation", source)
			})?;

		let words = match artifact.spirv_words() {
			Ok(words) => words,
			Err(error) => {
				destroy_pipeline_layout(device, layout);
				return Err(error);
			}
		};
		let module_info = ash::vk::ShaderModuleCreateInfo::default().code(&words);
		// SAFETY: build-time spirv-val and runtime word validation proved a complete
		// SPIR-V word stream; the device remains live through pipeline creation.
		let module = match unsafe { device.create_shader_module(&module_info, None) } {
			Ok(module) => module,
			Err(source) => {
				destroy_pipeline_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute shader-module creation",
					source,
				));
			}
		};

		let stage = ash::vk::PipelineShaderStageCreateInfo::default()
			.stage(ash::vk::ShaderStageFlags::COMPUTE)
			.module(module)
			.name(artifact.entry_point);
		let pipeline_info = ash::vk::ComputePipelineCreateInfo::default()
			.stage(stage)
			.layout(layout);
		// SAFETY: the module contains the reflected compute entry point, and the
		// pipeline layout exactly matches its validated descriptor and push ABI.
		let pipeline_result = unsafe {
			device.create_compute_pipelines(ash::vk::PipelineCache::null(), &[pipeline_info], None)
		};
		// The module is no longer needed after pipeline creation, successful or not.
		unsafe {
			device.destroy_shader_module(module, None);
		}
		let pipelines = match pipeline_result {
			Ok(pipelines) => pipelines,
			Err((partial, source)) => {
				for pipeline in partial {
					destroy_pipeline(device, pipeline);
				}
				destroy_pipeline_layout(device, layout);
				return Err(Error::backend_failure(
					"Vulkan",
					"compute-pipeline creation",
					source,
				));
			}
		};
		let Some(handle) = pipelines.first().copied() else {
			destroy_pipeline_layout(device, layout);
			return Err(Error::backend_failure(
				"Vulkan",
				"compute-pipeline creation",
				std::io::Error::other("Vulkan returned no pipeline after successful creation"),
			));
		};

		Ok(Self {
			handle,
			layout,
			max_dispatch_group_count: limits.max_compute_work_group_count,
			push_constant_size: artifact.push_constant_size,
		})
	}

	pub(super) fn destroy(&mut self, device: &ash::Device) {
		destroy_pipeline(device, self.handle);
		destroy_pipeline_layout(device, self.layout);
	}

	pub(super) const fn raw(&self) -> ash::vk::Pipeline {
		self.handle
	}

	pub(super) const fn layout(&self) -> ash::vk::PipelineLayout {
		self.layout
	}

	pub(super) const fn max_dispatch_group_count(&self) -> [u32; 3] {
		self.max_dispatch_group_count
	}

	pub(super) const fn push_constant_size(&self) -> u32 {
		self.push_constant_size
	}
}

fn validate_artifact_limits(
	workgroup_size: [u32; 3],
	push_constant_size: u32,
	limits: DeviceLimits,
) -> Result<()> {
	let invocation_count = workgroup_size
		.into_iter()
		.try_fold(1_u32, u32::checked_mul)
		.ok_or_else(|| Error::missing_capability("shader workgroup size overflows u32"))?;
	if push_constant_size > limits.max_push_constants_size
		|| invocation_count > limits.max_compute_work_group_invocations
		|| workgroup_size
			.into_iter()
			.zip(limits.max_compute_work_group_size)
			.any(|(required, available)| required > available)
	{
		return Err(Error::missing_capability(
			"selected device cannot admit the reflected compute shader ABI",
		));
	}
	Ok(())
}

fn destroy_pipeline(device: &ash::Device, pipeline: ash::vk::Pipeline) {
	// SAFETY: the handle belongs to `device` and is destroyed at most once after use.
	unsafe { device.destroy_pipeline(pipeline, None) }
}

fn destroy_pipeline_layout(device: &ash::Device, layout: ash::vk::PipelineLayout) {
	// SAFETY: the layout belongs to `device` and no live pipeline uses it.
	unsafe { device.destroy_pipeline_layout(layout, None) }
}
