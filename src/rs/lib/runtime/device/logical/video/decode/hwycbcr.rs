//! Donor: videoDecoderColor.cpp::convertNv12ToRgbHardwareIntoAsync.
//! Mechanical Vulkan ABI/RAII adaptation of immutable conversion/sampler,
//! hybrid descriptor sets and per-image/layer views. No color math on host.

use super::*;

struct SourceBinding {
	image: vk::Image,
	layer: u32,
	view: vk::ImageView,
	set: vk::DescriptorSet,
}

pub(in crate::runtime::device) struct HardwareYcbcrColorConversion {
	device: Arc<vk::Device>,
	_images: Arc<Vec<VideoImage>>,
	pub color_space: u32,
	pub full_range: bool,
	conversion: vk::SamplerYcbcrConversion,
	sampler: vk::Sampler,
	descriptors: vk::DescriptorSetLayout,
	pool: vk::DescriptorPool,
	module: vk::ShaderModule,
	layout: vk::PipelineLayout,
	pipeline: vk::Pipeline,
	sources: Vec<SourceBinding>,
}

pub(in crate::runtime::device) fn supported(device: &Device, format: vk::Format) -> bool {
	device.configuration().features().sampler_ycbcr_conversion
		&& device.optimal_format_supports(
			format,
			vk::FormatFeatureFlagBits::SAMPLED_IMAGE_BIT
				| vk::FormatFeatureFlagBits::MIDPOINT_CHROMA_SAMPLES_BIT,
		)
}

impl HardwareYcbcrColorConversion {
	#[cfg(oa_profile = "core")]
	pub(in crate::runtime::device) fn new(
		device: &Arc<Device>,
		images: Arc<Vec<VideoImage>>,
		color_space: u32,
		full_range: bool,
	) -> Result<Arc<Self>> {
		let limits = device.physical().limits;
		if limits.max_compute_work_group_invocations < 256
			|| limits.max_compute_work_group_size[0] < 16
			|| limits.max_compute_work_group_size[1] < 16
			|| limits.max_push_constants_size < 28
		{
			return Err(Error::missing_capability(
				"hardware YCbCr conversion exceeds compute ABI limits",
			));
		}
		let format = images
			.iter()
			.find(|image| image.sampled)
			.ok_or_else(|| Error::missing_capability("decoder images do not admit sampled use"))?
			.format;
		if !supported(device, format) {
			return Err(Error::missing_capability(
				"decoded format lacks midpoint hardware YCbCr sampling",
			));
		}
		let count = images
			.iter()
			.filter(|image| image.sampled)
			.try_fold(0_u32, |sum, image| sum.checked_add(image.array_layers))
			.ok_or_else(|| Error::out_of_range("YCbCr image count overflow"))?;
		let mut properties = vk::SamplerYcbcrConversionImageFormatProperties::new();
		let mut image_properties = vk::ImageFormatProperties2::new()
			.push_next(&mut properties)
			.map_err(Error::from)?;
		let query = vk::PhysicalDeviceImageFormatInfo2::new()
			.set_format(format)
			.set_type(vk::ImageType::_2D)
			.set_tiling(vk::ImageTiling::OPTIMAL)
			.set_usage(vk::ImageUsageFlagBits::SAMPLED_BIT);
		// SAFETY: the selected physical device and initialized output chain belong
		// to the retained instance; this queries sampled descriptor cost only.
		unsafe {
			device
				.instance()
				.raw()
				.get_physical_device_image_format_properties2(
					device.physical().handle,
					&query,
					&mut image_properties,
				)
		}
		.map_err(|e| Error::backend_failure("Vulkan", "YCbCr image-format query", e))?;
		let pool_count = count
			.checked_mul(properties.combined_image_sampler_descriptor_count.max(1))
			.ok_or_else(|| Error::out_of_range("YCbCr descriptor count overflow"))?;
		let mut state = Self {
			device: device.raw_arc(),
			_images: images,
			color_space,
			full_range,
			conversion: vk::SamplerYcbcrConversion::default(),
			sampler: vk::Sampler::default(),
			descriptors: vk::DescriptorSetLayout::default(),
			pool: vk::DescriptorPool::default(),
			module: vk::ShaderModule::default(),
			layout: vk::PipelineLayout::default(),
			pipeline: vk::Pipeline::default(),
			sources: Vec::new(),
		};
		let info = vk::SamplerYcbcrConversionCreateInfo::new()
			.set_format(format)
			.set_ycbcr_model(match color_space {
				0 => vk::SamplerYcbcrModelConversion::YCBCR_601,
				2 => vk::SamplerYcbcrModelConversion::YCBCR_2020,
				_ => vk::SamplerYcbcrModelConversion::YCBCR_709,
			})
			.set_ycbcr_range(if full_range {
				vk::SamplerYcbcrRange::ITU_FULL
			} else {
				vk::SamplerYcbcrRange::ITU_NARROW
			})
			.set_components(vk::ComponentMapping::default())
			.set_x_chroma_offset(vk::ChromaLocation::MIDPOINT)
			.set_y_chroma_offset(vk::ChromaLocation::MIDPOINT)
			.set_chroma_filter(vk::Filter::NEAREST);
		// SAFETY: feature, format and midpoint support were admitted above; all
		// resources are uniquely owned by state, whose Drop handles partial failure.
		unsafe {
			state.conversion = state
				.device
				.create_sampler_ycbcr_conversion(&info, None)
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr conversion creation", e))?;
			let mut conversion = vk::SamplerYcbcrConversionInfo::new().set_conversion(state.conversion);
			let info = vk::SamplerCreateInfo::new()
				.set_mag_filter(vk::Filter::NEAREST)
				.set_min_filter(vk::Filter::NEAREST)
				.set_mipmap_mode(vk::SamplerMipmapMode::NEAREST)
				.set_address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
				.set_address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
				.set_address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
				.push_next(&mut conversion)
				.map_err(Error::from)?;
			state.sampler = state
				.device
				.create_sampler(&info, None)
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr immutable sampler", e))?;
			let samplers = [state.sampler];
			let bindings = [vk::DescriptorSetLayoutBinding::new()
				.set_binding(0)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_descriptor_count(1)
				.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT)
				.set_immutable_samplers(&samplers)
				.map_err(Error::from)?];
			state.descriptors = state
				.device
				.create_descriptor_set_layout(
					&vk::DescriptorSetLayoutCreateInfo::new()
						.set_bindings(&bindings)
						.map_err(Error::from)?,
					None,
				)
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr descriptor layout", e))?;
			let sizes = [vk::DescriptorPoolSize::new()
				.set_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_descriptor_count(pool_count)];
			state.pool = state
				.device
				.create_descriptor_pool(
					&vk::DescriptorPoolCreateInfo::new()
						.set_max_sets(count)
						.set_pool_sizes(&sizes)
						.map_err(Error::from)?,
					None,
				)
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr descriptor pool", e))?;
			let layouts = [device.descriptor_layout(), state.descriptors];
			let pushes = [vk::PushConstantRange::new()
				.set_stage_flags(vk::ShaderStageFlagBits::COMPUTE_BIT)
				.set_size(28)];
			state.layout = state
				.device
				.create_pipeline_layout(
					&vk::PipelineLayoutCreateInfo::new()
						.set_set_layouts(&layouts)
						.map_err(Error::from)?
						.set_push_constant_ranges(&pushes)
						.map_err(Error::from)?,
					None,
				)
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr hybrid pipeline layout", e))?;
			let words = crate::runtime::decode_spirv_bytes(include_bytes!(concat!(
				env!("OA_SHADER_DIR"),
				"/video_cvt_ycbcr_to_rgba8.spv"
			)))?;
			state.module = state
				.device
				.create_shader_module(&vk::ShaderModuleCreateInfo::new().set_code(&words), None)
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr shader module", e))?;
			let stage = vk::PipelineShaderStageCreateInfo::new()
				.set_stage(vk::ShaderStageFlagBits::COMPUTE_BIT)
				.set_module(state.module)
				.set_name(c"main");
			let info = vk::ComputePipelineCreateInfo::new()
				.set_stage(stage)
				.set_layout(state.layout);
			let mut pipelines = [vk::Pipeline::default()];
			let result = state.device.create_compute_pipelines(
				vk::PipelineCache::default(),
				&[info],
				None,
				&mut pipelines,
			);
			state.pipeline = pipelines[0];
			if !matches!(result, Ok(vk::Result::SUCCESS)) {
				return Err(Error::backend_dispatch_failure(
					"Vulkan",
					"YCbCr compute pipeline",
					result,
				));
			}
		}
		for image in state._images.iter().filter(|image| image.sampled) {
			for layer in 0..image.array_layers {
				let mut conversion = vk::SamplerYcbcrConversionInfo::new().set_conversion(state.conversion);
				let info = vk::ImageViewCreateInfo::new()
					.set_image(image.handle)
					.set_view_type(vk::ImageViewType::_2D)
					.set_format(format)
					.set_components(vk::ComponentMapping::default())
					.set_subresource_range(decode_image_subresource(layer, 1))
					.push_next(&mut conversion)
					.map_err(Error::from)?;
				// SAFETY: each sampled image layer is bounded and retained by state;
				// the view's conversion exactly matches the immutable sampler.
				let view = unsafe { state.device.create_image_view(&info, None) }
					.map_err(|e| Error::backend_failure("Vulkan", "YCbCr source view", e))?;
				state.sources.push(SourceBinding {
					image: image.handle,
					layer,
					view,
					set: vk::DescriptorSet::default(),
				});
			}
		}
		let layouts = vec![state.descriptors; state.sources.len()];
		let mut sets = vec![vk::DescriptorSet::default(); layouts.len()];
		// SAFETY: the pool includes the queried multi-plane descriptor cost for
		// every immutable source set. No submitted descriptors are ever mutated.
		unsafe {
			state.device.allocate_descriptor_sets(
				&vk::DescriptorSetAllocateInfo::new()
					.set_descriptor_pool(state.pool)
					.set_set_layouts(&layouts)
					.map_err(Error::from)?,
				&mut sets,
			)
		}
		.map_err(|e| Error::backend_failure("Vulkan", "YCbCr source sets", e))?;
		for (source, set) in state.sources.iter_mut().zip(sets) {
			source.set = set;
			let image = [vk::DescriptorImageInfo::new()
				.set_sampler(state.sampler)
				.set_image_view(source.view)
				.set_image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
			let writes = [vk::WriteDescriptorSet::new()
				.set_dst_set(set)
				.set_dst_binding(0)
				.set_descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
				.set_image_info(&image)
				.map_err(Error::from)?];
			// SAFETY: the newly allocated set is not in flight; view and immutable
			// sampler remain alive until every consumer has completed.
			unsafe { state.device.update_descriptor_sets(&writes, &[]) }
				.map_err(|e| Error::backend_failure("Vulkan", "YCbCr source binding", e))?;
		}
		crate::log_info!(
			crate::LogComponent::VIDEO,
			"video presentation: hardware YCbCr, format {:?}, matrix {}, full-range {} (direct decoded-image sampling)",
			format,
			color_space,
			full_range
		);
		Ok(Arc::new(state))
	}

	#[cfg(oa_profile = "compatibility")]
	pub(in crate::runtime::device) fn new(
		_device: &Arc<Device>,
		_images: Arc<Vec<VideoImage>>,
		_color_space: u32,
		_full_range: bool,
	) -> Result<Arc<Self>> {
		Err(Error::missing_capability(
			"native video presentation requires the Core profile",
		))
	}

	pub(in crate::runtime::device) fn record(
		&self,
		device: &Device,
		command: vk::CommandBuffer,
		image: &VideoImage,
		layer: u32,
		rgba: &crate::runtime::Buffer,
		session: &DecodeSession,
	) -> Result<()> {
		let source = self
			.sources
			.iter()
			.find(|source| source.image == image.handle && source.layer == layer)
			.ok_or_else(|| Error::invalid_argument("YCbCr frame has no retained source view"))?;
		let extent = session.visible_extent;
		let max = device.physical().limits.max_compute_work_group_count;
		if extent.width.div_ceil(16) > max[0] || extent.height.div_ceil(16) > max[1] {
			return Err(Error::missing_capability(
				"YCbCr dispatch exceeds device limits",
			));
		}
		let push = [
			rgba.descriptor_index(),
			extent.width,
			extent.height,
			session.coded_extent.width,
			session.coded_extent.height,
			session.visible_origin[0],
			session.visible_origin[1],
		];
		// SAFETY: the active compute command, immutable pipeline/set and output
		// belong to device. The caller acquired this layer for sampled reads and
		// retains self and rgba through the exact conversion completion.
		unsafe {
			self
				.device
				.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline)?;
			self.device.cmd_bind_descriptor_sets(
				command,
				vk::PipelineBindPoint::COMPUTE,
				self.layout,
				0,
				&[device.descriptor_set(), source.set],
				&[],
			)?;
			self.device.cmd_push_constants(
				command,
				self.layout,
				vk::ShaderStageFlagBits::COMPUTE_BIT,
				0,
				std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), size_of_val(&push)),
			)?;
			self.device.cmd_dispatch(
				command,
				extent.width.div_ceil(16),
				extent.height.div_ceil(16),
				1,
			)?;
		}
		Ok(())
	}
}

impl Drop for HardwareYcbcrColorConversion {
	fn drop(&mut self) {
		// SAFETY: partial construction has never submitted; a published Arc is
		// retained through its last conversion event. Views die before images,
		// pipelines/pools before layouts/samplers, and conversion before Device.
		unsafe {
			if !self.pipeline.is_null() {
				let _ = self.device.destroy_pipeline(self.pipeline, None);
			}
			if !self.module.is_null() {
				let _ = self.device.destroy_shader_module(self.module, None);
			}
			if !self.layout.is_null() {
				let _ = self.device.destroy_pipeline_layout(self.layout, None);
			}
			if !self.pool.is_null() {
				let _ = self.device.destroy_descriptor_pool(self.pool, None);
			}
			for source in &self.sources {
				let _ = self.device.destroy_image_view(source.view, None);
			}
			if !self.descriptors.is_null() {
				let _ = self
					.device
					.destroy_descriptor_set_layout(self.descriptors, None);
			}
			if !self.sampler.is_null() {
				let _ = self.device.destroy_sampler(self.sampler, None);
			}
			if !self.conversion.is_null() {
				let _ = self
					.device
					.destroy_sampler_ycbcr_conversion(self.conversion, None);
			}
		}
	}
}

impl DecodeSession {
	pub(in crate::runtime::device) fn record_hardware_presentation(
		&mut self,
		device: &Arc<Device>,
		slot: u32,
		layer: u32,
		color_space: u32,
		full_range: bool,
		frame: &NativeDecodedFrame,
	) -> Result<NativePresentation> {
		let pipeline = if let Some(pipeline) = self
			.hardware_ycbcr
			.iter()
			.find(|pipeline| pipeline.color_space == color_space && pipeline.full_range == full_range)
		{
			pipeline.clone()
		} else {
			let pipeline =
				HardwareYcbcrColorConversion::new(device, self.images.clone(), color_space, full_range)?;
			self.hardware_ycbcr.push(pipeline.clone());
			pipeline
		};
		let extent = self.visible_extent;
		let bytes = usize::try_from(extent.width)
			.ok()
			.and_then(|w| w.checked_mul(extent.height as usize))
			.and_then(|n| n.checked_mul(4))
			.ok_or_else(|| Error::out_of_range("YCbCr presentation extent overflow"))?;
		let rgba = crate::runtime::Buffer::host_visible_storage(device, bytes)?;
		let families = (
			device
				.physical()
				.video
				.decode_queue_family
				.ok_or_else(|| Error::missing_capability("video decode queue is unavailable"))?,
			device.physical().compute_queue_family,
		);
		let layout = if self.image_set == DecodeImageSet::DistinctLayered {
			vk::ImageLayout::VIDEO_DECODE_DST_KHR
		} else {
			vk::ImageLayout::VIDEO_DECODE_DPB_KHR
		};
		let (output, _, _, _) = self.picture_images(slot)?;
		let mut commands = record_image_consumer(
			device,
			output,
			layer,
			families.0,
			layout,
			ImageUse::sample(families.1),
			|command| {
				pipeline.record(device, command, output, layer, &rgba, self)?;
				let ready = vk::BufferMemoryBarrier2::new()
					.set_src_stage_mask(vk::PipelineStageFlagBits2::COMPUTE_SHADER_BIT)
					.set_src_access_mask(vk::AccessFlagBits2::SHADER_STORAGE_WRITE_BIT)
					.set_dst_stage_mask(
						vk::PipelineStageFlagBits2::TRANSFER_BIT | vk::PipelineStageFlagBits2::HOST_BIT,
					)
					.set_dst_access_mask(
						vk::AccessFlagBits2::TRANSFER_READ_BIT | vk::AccessFlagBits2::HOST_READ_BIT,
					)
					.set_src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.set_dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
					.set_buffer(rgba.raw())
					.set_offset(0)
					.set_size(rgba.size());
				let dependency = vk::DependencyInfo::new()
					.set_buffer_memory_barriers(std::slice::from_ref(&ready))
					.map_err(Error::from)?;
				// SAFETY: publish exactly this dispatch's RGBA writes to subsequent
				// Texture upload or explicit host observation after its completion.
				unsafe { device.cmd_pipeline_barrier2(command, &dependency) }?;
				Ok(())
			},
		)?;
		commands.retain(Arc::new((frame.clone(), rgba.clone(), pipeline)));
		Ok(NativePresentation {
			commands,
			rgba,
			extent,
		})
	}
}
