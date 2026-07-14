#include <Oa/Render/Renderer.h>

#include <Oa/Render/FnMesh.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/OaVma.h>
#include <Oa/Runtime/ShaderProvider.h>
#include <cstddef>
#include <cstring>

namespace {

struct CanvasPushConstants {
	VlmMat4 Mvp;
	VlmVec4 Tint;
};
static_assert(sizeof(CanvasPushConstants) == 80);

struct DetectionPushConstants {
	VlmMat4 Mvp;
	VlmVec4 Color;
	OaU32 DetectionIndex;
	OaU32 TargetWidth;
	OaU32 TargetHeight;
	OaF32 ThicknessPixels;
};
static_assert(sizeof(DetectionPushConstants) == 96);

struct GlyphPushConstants {
	VlmMat4 Mvp;
	OaU32 GlyphIndex;
	OaU32 AtlasIndex;
	OaU32 ReferenceWidth;
	OaU32 ReferenceHeight;
	OaU32 AtlasWidth;
	OaU32 AtlasHeight;
	OaF32 PxRange;
	OaU32 Reserved0;
};
static_assert(sizeof(GlyphPushConstants) == 96);

VkShaderModule CreateShaderModule(VkDevice InDevice, const OaSpvEntry& InSpv) {
	VkShaderModuleCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = InSpv.Size;
	info.pCode = reinterpret_cast<const OaU32*>(InSpv.Data);
	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(InDevice, &info, nullptr, &module) != VK_SUCCESS) {
		return VK_NULL_HANDLE;
	}
	return module;
}

OaResult<OaVkBuffer> CreateMappedRenderBuffer(
	OaGraphicsEngine& InEngine,
	OaU64 InSize,
	VkBufferUsageFlags InUsage) {
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = InSize;
	bufferInfo.usage = InUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	OaVmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = OA_VMA_MEMORY_USAGE_CPU_TO_GPU;
	allocationInfo.flags =
		OA_VMA_ALLOCATION_CREATE_MAPPED_BIT |
		OA_VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

	VkBuffer buffer = VK_NULL_HANDLE;
	OaVmaAllocation allocation = VK_NULL_HANDLE;
	OaVmaAllocationInfo mappedInfo{};
	if (OaVmaCreateBuffer(
		static_cast<OaVmaAllocator>(InEngine.Allocator.Allocator),
		&bufferInfo, &allocationInfo, &buffer, &allocation,
		&mappedInfo) != VK_SUCCESS) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaCanvasRenderer: render buffer allocation failed");
	}

	OaVkBuffer result;
	result.Buffer = buffer;
	result.Allocation = allocation;
	result.Size = InSize;
	result.MappedPtr = mappedInfo.pMappedData;
	return result;
}

} // namespace

struct OaCanvasRenderer::Impl {
	OaGraphicsEngine* Engine = nullptr;
	VkDescriptorSetLayout DescriptorSetLayout = VK_NULL_HANDLE;
	VkPipelineLayout PipelineLayout = VK_NULL_HANDLE;
	VkPipelineLayout DetectionPipelineLayout = VK_NULL_HANDLE;
	VkPipelineLayout GlyphPipelineLayout = VK_NULL_HANDLE;
	VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
	VkPipeline Pipeline = VK_NULL_HANDLE;
	VkPipeline DetectionPipeline = VK_NULL_HANDLE;
	VkPipeline GlyphPipeline = VK_NULL_HANDLE;
	VkSampler Sampler = VK_NULL_HANDLE;
	OaVkBuffer VertexBuffer;
	OaVkBuffer IndexBuffer;
	VkImage TargetImage = VK_NULL_HANDLE;
	VkImageView TargetView = VK_NULL_HANDLE;
	OaVmaAllocation TargetAllocation = VK_NULL_HANDLE;
	VkImageLayout TargetLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	OaU32 TargetWidth = 0;
	OaU32 TargetHeight = 0;
	OaVec<OaCanvasImageDraw> Draws;
	OaVec<OaCanvasRectInstanceDraw> DetectionDraws;
	OaVec<OaCanvasGlyphInstanceDraw> GlyphDraws;

	struct SampledImage {
		VkImageView View = VK_NULL_HANDLE;
		VkImageLayout Layout = VK_IMAGE_LAYOUT_UNDEFINED;
		VkDescriptorSet Set = VK_NULL_HANDLE;
	};
	OaVec<SampledImage> SampledImages;
};

static void DestroyTarget(OaCanvasRenderer::Impl& InImpl) {
	if (!InImpl.Engine) return;
	VkDevice device = static_cast<VkDevice>(InImpl.Engine->Device.Device);
	if (InImpl.TargetView) {
		vkDestroyImageView(device, InImpl.TargetView, nullptr);
	}
	if (InImpl.TargetImage) {
		OaVmaDestroyImage(
			static_cast<OaVmaAllocator>(InImpl.Engine->Allocator.Allocator),
			InImpl.TargetImage, InImpl.TargetAllocation);
	}
	InImpl.TargetImage = VK_NULL_HANDLE;
	InImpl.TargetView = VK_NULL_HANDLE;
	InImpl.TargetAllocation = VK_NULL_HANDLE;
	InImpl.TargetLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	InImpl.TargetWidth = 0;
	InImpl.TargetHeight = 0;
}

static OaStatus CreateTarget(
	OaCanvasRenderer::Impl& InImpl,
	OaU32 InWidth,
	OaU32 InHeight) {
	if (InWidth == 0 || InHeight == 0) {
		return OaStatus::Error(
			OaStatusCode::InvalidArgument,
			"OaCanvasRenderer: target dimensions must be non-zero");
	}
	DestroyTarget(InImpl);

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.extent = {InWidth, InHeight, 1};
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage =
		VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
		VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	OaVmaAllocationCreateInfo allocationInfo{};
	allocationInfo.usage = OA_VMA_MEMORY_USAGE_GPU_ONLY;
	if (OaVmaCreateImage(
		static_cast<OaVmaAllocator>(InImpl.Engine->Allocator.Allocator),
		&imageInfo, &allocationInfo, &InImpl.TargetImage,
		&InImpl.TargetAllocation, nullptr) != VK_SUCCESS) {
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaCanvasRenderer: target image allocation failed");
	}

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = InImpl.TargetImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = imageInfo.format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.layerCount = 1;
	VkDevice device =
		static_cast<VkDevice>(InImpl.Engine->Device.Device);
	if (vkCreateImageView(
		device, &viewInfo, nullptr, &InImpl.TargetView) != VK_SUCCESS) {
		DestroyTarget(InImpl);
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			"OaCanvasRenderer: target image view creation failed");
	}
	InImpl.TargetWidth = InWidth;
	InImpl.TargetHeight = InHeight;
	return OaStatus::Ok();
}

OaCanvasRenderer::OaCanvasRenderer(OaCanvasRenderer&& InOther) noexcept
	: Impl_(OaStdMove(InOther.Impl_)) {}

OaCanvasRenderer& OaCanvasRenderer::operator=(
	OaCanvasRenderer&& InOther) noexcept {
	if (this != &InOther) {
		Destroy();
		Impl_ = OaStdMove(InOther.Impl_);
	}
	return *this;
}

OaCanvasRenderer::~OaCanvasRenderer() { Destroy(); }

OaStatus OaCanvasRenderer::Init(
	OaGraphicsEngine& InEngine,
	OaU32 InTargetWidth,
	OaU32 InTargetHeight) {
	Destroy();
	Impl_ = OaStdMakeUnique<Impl>();
	Impl_->Engine = &InEngine;
	VkDevice device = static_cast<VkDevice>(InEngine.Device.Device);

	VkDescriptorSetLayoutBinding resourceBindings[2]{};
	resourceBindings[0].binding = 0;
	resourceBindings[0].descriptorType =
		VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
	resourceBindings[0].descriptorCount = 1;
	resourceBindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	resourceBindings[1].binding = 1;
	resourceBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
	resourceBindings[1].descriptorCount = 1;
	resourceBindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	VkDescriptorSetLayoutCreateInfo setLayoutInfo{};
	setLayoutInfo.sType =
		VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setLayoutInfo.bindingCount = 2;
	setLayoutInfo.pBindings = resourceBindings;
	if (vkCreateDescriptorSetLayout(
		device, &setLayoutInfo, nullptr,
		&Impl_->DescriptorSetLayout) != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			"OaCanvasRenderer: descriptor set layout creation failed");
	}

	VkPushConstantRange pushRange{};
	pushRange.stageFlags =
		VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	pushRange.size = sizeof(CanvasPushConstants);
	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType =
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &Impl_->DescriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushRange;
	if (vkCreatePipelineLayout(
		device, &pipelineLayoutInfo, nullptr,
		&Impl_->PipelineLayout) != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: pipeline layout creation failed");
	}

	VkDescriptorPoolSize poolSizes[2] = {
		{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256},
		{VK_DESCRIPTOR_TYPE_SAMPLER, 256},
	};
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 256;
	poolInfo.poolSizeCount = 2;
	poolInfo.pPoolSizes = poolSizes;
	if (vkCreateDescriptorPool(
		device, &poolInfo, nullptr, &Impl_->DescriptorPool) != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::ResourceExhausted,
			"OaCanvasRenderer: descriptor pool creation failed");
	}

	const OaSpvEntry* vert = OaShaderProviderFind("UnlitTextured.vert");
	const OaSpvEntry* frag = OaShaderProviderFind("UnlitTextured.frag");
	if (!vert || !frag) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::NotFound,
			"OaCanvasRenderer: shaders not found");
	}
	VkShaderModule vertModule = CreateShaderModule(device, *vert);
	VkShaderModule fragModule = CreateShaderModule(device, *frag);
	if (!vertModule || !fragModule) {
		if (vertModule) vkDestroyShaderModule(device, vertModule, nullptr);
		if (fragModule) vkDestroyShaderModule(device, fragModule, nullptr);
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: shader module creation failed");
	}

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType =
		VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vertModule;
	stages[0].pName = "main";
	stages[1].sType =
		VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = fragModule;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{
		0, sizeof(OaMeshVertex), VK_VERTEX_INPUT_RATE_VERTEX};
	VkVertexInputAttributeDescription attributes[2]{};
	attributes[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,
		static_cast<OaU32>(offsetof(OaMeshVertex, Position))};
	attributes[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,
		static_cast<OaU32>(offsetof(OaMeshVertex, Uv))};
	VkPipelineVertexInputStateCreateInfo vertexInput{};
	vertexInput.sType =
		VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInput.vertexBindingDescriptionCount = 1;
	vertexInput.pVertexBindingDescriptions = &binding;
	vertexInput.vertexAttributeDescriptionCount = 2;
	vertexInput.pVertexAttributeDescriptions = attributes;
	VkPipelineInputAssemblyStateCreateInfo assembly{};
	assembly.sType =
		VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo viewport{};
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1;
	viewport.scissorCount = 1;
	VkPipelineRasterizationStateCreateInfo raster{};
	raster.sType =
		VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	raster.polygonMode = VK_POLYGON_MODE_FILL;
	raster.cullMode = VK_CULL_MODE_NONE;
	raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	raster.lineWidth = 1.0F;
	VkPipelineMultisampleStateCreateInfo multisample{};
	multisample.sType =
		VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineColorBlendAttachmentState blendAttachment{};
	blendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkPipelineColorBlendStateCreateInfo blend{};
	blend.sType =
		VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	blend.attachmentCount = 1;
	blend.pAttachments = &blendAttachment;
	const VkDynamicState dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType =
		VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamicStates;
	VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
	VkPipelineRenderingCreateInfo rendering{};
	rendering.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachmentFormats = &colorFormat;
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType =
		VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineInfo.pNext = &rendering;
	pipelineInfo.stageCount = 2;
	pipelineInfo.pStages = stages;
	pipelineInfo.pVertexInputState = &vertexInput;
	pipelineInfo.pInputAssemblyState = &assembly;
	pipelineInfo.pViewportState = &viewport;
	pipelineInfo.pRasterizationState = &raster;
	pipelineInfo.pMultisampleState = &multisample;
	pipelineInfo.pColorBlendState = &blend;
	pipelineInfo.pDynamicState = &dynamic;
	pipelineInfo.layout = Impl_->PipelineLayout;
	const VkResult pipelineResult = vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
		&Impl_->Pipeline);
	vkDestroyShaderModule(device, vertModule, nullptr);
	vkDestroyShaderModule(device, fragModule, nullptr);
	if (pipelineResult != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: graphics pipeline creation failed");
	}

	const OaSpvEntry* detectionVert =
		OaShaderProviderFind("DetectionOverlay.vert");
	const OaSpvEntry* detectionFrag =
		OaShaderProviderFind("DetectionOverlay.frag");
	if (!detectionVert || !detectionFrag) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::NotFound,
			"OaCanvasRenderer: detection shaders not found");
	}
	VkShaderModule detectionVertModule =
		CreateShaderModule(device, *detectionVert);
	VkShaderModule detectionFragModule =
		CreateShaderModule(device, *detectionFrag);
	if (!detectionVertModule || !detectionFragModule) {
		if (detectionVertModule) {
			vkDestroyShaderModule(device, detectionVertModule, nullptr);
		}
		if (detectionFragModule) {
			vkDestroyShaderModule(device, detectionFragModule, nullptr);
		}
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: detection shader module creation failed");
	}
	VkPushConstantRange detectionPushRange{};
	detectionPushRange.stageFlags =
		VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	detectionPushRange.size = sizeof(DetectionPushConstants);
	const VkDescriptorSetLayout bindlessLayout =
		static_cast<VkDescriptorSetLayout>(
			InEngine.Bindless.DescriptorSetLayout);
	VkPipelineLayoutCreateInfo detectionLayoutInfo{};
	detectionLayoutInfo.sType =
		VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	detectionLayoutInfo.setLayoutCount = 1;
	detectionLayoutInfo.pSetLayouts = &bindlessLayout;
	detectionLayoutInfo.pushConstantRangeCount = 1;
	detectionLayoutInfo.pPushConstantRanges = &detectionPushRange;
	if (vkCreatePipelineLayout(
		device, &detectionLayoutInfo, nullptr,
		&Impl_->DetectionPipelineLayout) != VK_SUCCESS) {
		vkDestroyShaderModule(device, detectionVertModule, nullptr);
		vkDestroyShaderModule(device, detectionFragModule, nullptr);
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: detection pipeline layout creation failed");
	}
	VkPipelineShaderStageCreateInfo detectionStages[2] = {
		stages[0], stages[1]};
	detectionStages[0].module = detectionVertModule;
	detectionStages[1].module = detectionFragModule;
	VkPipelineVertexInputStateCreateInfo noVertexInput{};
	noVertexInput.sType =
		VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	VkPipelineColorBlendAttachmentState detectionBlendAttachment =
		blendAttachment;
	detectionBlendAttachment.blendEnable = VK_TRUE;
	detectionBlendAttachment.srcColorBlendFactor =
		VK_BLEND_FACTOR_SRC_ALPHA;
	detectionBlendAttachment.dstColorBlendFactor =
		VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	detectionBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
	detectionBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
	detectionBlendAttachment.dstAlphaBlendFactor =
		VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
	detectionBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
	VkPipelineColorBlendStateCreateInfo detectionBlend = blend;
	detectionBlend.pAttachments = &detectionBlendAttachment;
	VkGraphicsPipelineCreateInfo detectionPipelineInfo = pipelineInfo;
	detectionPipelineInfo.pStages = detectionStages;
	detectionPipelineInfo.pVertexInputState = &noVertexInput;
	detectionPipelineInfo.pColorBlendState = &detectionBlend;
	detectionPipelineInfo.layout = Impl_->DetectionPipelineLayout;
	const VkResult detectionPipelineResult = vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &detectionPipelineInfo, nullptr,
		&Impl_->DetectionPipeline);
	vkDestroyShaderModule(device, detectionVertModule, nullptr);
	vkDestroyShaderModule(device, detectionFragModule, nullptr);
	if (detectionPipelineResult != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: detection graphics pipeline creation failed");
	}

	const OaSpvEntry* glyphVert =
		OaShaderProviderFind("CanvasGlyph.vert");
	const OaSpvEntry* glyphFrag =
		OaShaderProviderFind("CanvasGlyph.frag");
	if (!glyphVert || !glyphFrag) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::NotFound,
			"OaCanvasRenderer: glyph shaders not found");
	}
	VkShaderModule glyphVertModule = CreateShaderModule(device, *glyphVert);
	VkShaderModule glyphFragModule = CreateShaderModule(device, *glyphFrag);
	if (!glyphVertModule || !glyphFragModule) {
		if (glyphVertModule) vkDestroyShaderModule(device, glyphVertModule, nullptr);
		if (glyphFragModule) vkDestroyShaderModule(device, glyphFragModule, nullptr);
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: glyph shader module creation failed");
	}
	VkPushConstantRange glyphPushRange{};
	glyphPushRange.stageFlags =
		VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	glyphPushRange.size = sizeof(GlyphPushConstants);
	VkPipelineLayoutCreateInfo glyphLayoutInfo = detectionLayoutInfo;
	glyphLayoutInfo.pPushConstantRanges = &glyphPushRange;
	if (vkCreatePipelineLayout(
		device, &glyphLayoutInfo, nullptr,
		&Impl_->GlyphPipelineLayout) != VK_SUCCESS) {
		vkDestroyShaderModule(device, glyphVertModule, nullptr);
		vkDestroyShaderModule(device, glyphFragModule, nullptr);
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: glyph pipeline layout creation failed");
	}
	VkPipelineShaderStageCreateInfo glyphStages[2] = {
		stages[0], stages[1]};
	glyphStages[0].module = glyphVertModule;
	glyphStages[1].module = glyphFragModule;
	VkGraphicsPipelineCreateInfo glyphPipelineInfo = detectionPipelineInfo;
	glyphPipelineInfo.pStages = glyphStages;
	glyphPipelineInfo.layout = Impl_->GlyphPipelineLayout;
	const VkResult glyphPipelineResult = vkCreateGraphicsPipelines(
		device, VK_NULL_HANDLE, 1, &glyphPipelineInfo, nullptr,
		&Impl_->GlyphPipeline);
	vkDestroyShaderModule(device, glyphVertModule, nullptr);
	vkDestroyShaderModule(device, glyphFragModule, nullptr);
	if (glyphPipelineResult != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::PipelineError,
			"OaCanvasRenderer: glyph graphics pipeline creation failed");
	}

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	if (vkCreateSampler(
		device, &samplerInfo, nullptr, &Impl_->Sampler) != VK_SUCCESS) {
		Destroy();
		return OaStatus::Error(
			OaStatusCode::VulkanError,
			"OaCanvasRenderer: sampler creation failed");
	}

	OaMeshData quad = OaFnMesh::CreateQuad(1.0F, 1.0F);
	auto vertices = CreateMappedRenderBuffer(
		InEngine, quad.Vertices.Size() * sizeof(OaMeshVertex),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	auto indices = CreateMappedRenderBuffer(
		InEngine, quad.Indices.Size() * sizeof(OaU32),
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	if (!vertices.IsOk() || !indices.IsOk()) {
		if (vertices.IsOk()) InEngine.Allocator.Free(*vertices);
		if (indices.IsOk()) InEngine.Allocator.Free(*indices);
		Destroy();
		return OaStatus::Error(
			OaStatusCode::OutOfMemory,
			"OaCanvasRenderer: mesh allocation failed");
	}
	Impl_->VertexBuffer = OaStdMove(*vertices);
	Impl_->IndexBuffer = OaStdMove(*indices);
	std::memcpy(
		Impl_->VertexBuffer.MappedPtr, quad.Vertices.Data(),
		quad.Vertices.Size() * sizeof(OaMeshVertex));
	std::memcpy(
		Impl_->IndexBuffer.MappedPtr, quad.Indices.Data(),
		quad.Indices.Size() * sizeof(OaU32));
	return CreateTarget(*Impl_, InTargetWidth, InTargetHeight);
}

void OaCanvasRenderer::SetTarget(
	OaU32 InTargetWidth,
	OaU32 InTargetHeight) {
	if (!Impl_ ||
		(Impl_->TargetWidth == InTargetWidth &&
		 Impl_->TargetHeight == InTargetHeight)) {
		return;
	}
	(void)CreateTarget(*Impl_, InTargetWidth, InTargetHeight);
}

void OaCanvasRenderer::Destroy() {
	if (!Impl_) return;
	if (Impl_->Engine) {
		auto& engine = *Impl_->Engine;
		VkDevice device = static_cast<VkDevice>(engine.Device.Device);
		DestroyTarget(*Impl_);
		if (Impl_->Sampler) {
			vkDestroySampler(device, Impl_->Sampler, nullptr);
		}
		if (Impl_->Pipeline) {
			vkDestroyPipeline(device, Impl_->Pipeline, nullptr);
		}
		if (Impl_->DetectionPipeline) {
			vkDestroyPipeline(device, Impl_->DetectionPipeline, nullptr);
		}
		if (Impl_->GlyphPipeline) {
			vkDestroyPipeline(device, Impl_->GlyphPipeline, nullptr);
		}
		if (Impl_->DescriptorPool) {
			vkDestroyDescriptorPool(
				device, Impl_->DescriptorPool, nullptr);
		}
		if (Impl_->PipelineLayout) {
			vkDestroyPipelineLayout(
				device, Impl_->PipelineLayout, nullptr);
		}
		if (Impl_->DetectionPipelineLayout) {
			vkDestroyPipelineLayout(
				device, Impl_->DetectionPipelineLayout, nullptr);
		}
		if (Impl_->GlyphPipelineLayout) {
			vkDestroyPipelineLayout(
				device, Impl_->GlyphPipelineLayout, nullptr);
		}
		if (Impl_->DescriptorSetLayout) {
			vkDestroyDescriptorSetLayout(
				device, Impl_->DescriptorSetLayout, nullptr);
		}
		if (Impl_->VertexBuffer.Buffer) {
			engine.Allocator.Free(Impl_->VertexBuffer);
		}
		if (Impl_->IndexBuffer.Buffer) {
			engine.Allocator.Free(Impl_->IndexBuffer);
		}
	}
	Impl_.Reset();
}

void OaCanvasRenderer::BeginFrame() {
	if (Impl_) {
		Impl_->Draws.Clear();
		Impl_->DetectionDraws.Clear();
		Impl_->GlyphDraws.Clear();
	}
}

void OaCanvasRenderer::DrawImage(const OaCanvasImageDraw& InDraw) {
	if (!Impl_ || !InDraw.SourceView || InDraw.SourceWidth == 0 ||
		InDraw.SourceHeight == 0) {
		return;
	}
	Impl_->Draws.PushBack(InDraw);
}

void OaCanvasRenderer::DrawRectInstances(
	const OaCanvasRectInstanceDraw& InDraw) {
	if (!Impl_ || InDraw.InstanceBufferIndex == UINT32_MAX ||
		InDraw.InstanceCount == 0) {
		return;
	}
	Impl_->DetectionDraws.PushBack(InDraw);
}

void OaCanvasRenderer::DrawGlyphInstances(
	const OaCanvasGlyphInstanceDraw& InDraw) {
	if (!Impl_ || InDraw.InstanceCount == 0
		|| InDraw.InstanceBufferIndex == UINT32_MAX
		|| InDraw.AtlasBufferIndex == UINT32_MAX
		|| InDraw.AtlasWidth == 0 || InDraw.AtlasHeight == 0) {
		return;
	}
	if (InDraw.ReferenceWidth == 0 || InDraw.ReferenceHeight == 0) return;
	Impl_->GlyphDraws.PushBack(InDraw);
}

bool OaCanvasRenderer::HasDraws() const noexcept {
	return Impl_ &&
		(!Impl_->Draws.Empty() || !Impl_->DetectionDraws.Empty()
			|| !Impl_->GlyphDraws.Empty());
}

void OaCanvasRenderer::Record(
	VkCommandBuffer InCommandBuffer,
	VkImage InDestinationImage) {
	if (!HasDraws() || !Impl_->Pipeline || !Impl_->TargetView ||
		!InDestinationImage) {
		return;
	}

	auto descriptorSet =
		[&](const OaCanvasImageDraw& InDraw) -> VkDescriptorSet {
		for (const auto& image : Impl_->SampledImages) {
			if (image.View == InDraw.SourceView &&
				image.Layout == InDraw.SourceLayout) {
				return image.Set;
			}
		}
		VkDescriptorSetAllocateInfo allocateInfo{};
		allocateInfo.sType =
			VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocateInfo.descriptorPool = Impl_->DescriptorPool;
		allocateInfo.descriptorSetCount = 1;
		allocateInfo.pSetLayouts = &Impl_->DescriptorSetLayout;
		VkDescriptorSet set = VK_NULL_HANDLE;
		VkDevice device =
			static_cast<VkDevice>(Impl_->Engine->Device.Device);
		if (vkAllocateDescriptorSets(
			device, &allocateInfo, &set) != VK_SUCCESS) {
			return static_cast<VkDescriptorSet>(VK_NULL_HANDLE);
		}
		VkDescriptorImageInfo imageInfo{};
		imageInfo.imageView = InDraw.SourceView;
		imageInfo.imageLayout = InDraw.SourceLayout;
		VkDescriptorImageInfo samplerInfo{};
		samplerInfo.sampler = Impl_->Sampler;
		VkWriteDescriptorSet writes[2]{};
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = set;
		writes[0].dstBinding = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		writes[0].pImageInfo = &imageInfo;
		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet = set;
		writes[1].dstBinding = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
		writes[1].pImageInfo = &samplerInfo;
		vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
		Impl_->SampledImages.PushBack({
			InDraw.SourceView, InDraw.SourceLayout, set});
		return set;
	};

	VkImageSubresourceRange range{
		VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	VkImageMemoryBarrier toAttachment{};
	toAttachment.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	toAttachment.oldLayout = Impl_->TargetLayout;
	toAttachment.newLayout =
		VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	toAttachment.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachment.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toAttachment.image = Impl_->TargetImage;
	toAttachment.subresourceRange = range;
	toAttachment.srcAccessMask =
		Impl_->TargetLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			? VK_ACCESS_TRANSFER_READ_BIT : 0;
	toAttachment.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	vkCmdPipelineBarrier(
		InCommandBuffer,
		Impl_->TargetLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
			? VK_PIPELINE_STAGE_TRANSFER_BIT
			: VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		0, 0, nullptr, 0, nullptr, 1, &toAttachment);

	VkClearValue clear{};
	clear.color.float32[0] = 0.07F;
	clear.color.float32[1] = 0.07F;
	clear.color.float32[2] = 0.07F;
	clear.color.float32[3] = 1.0F;
	VkRenderingAttachmentInfo color{};
	color.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	color.imageView = Impl_->TargetView;
	color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	color.clearValue = clear;
	VkRenderingInfo rendering{};
	rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	rendering.renderArea.extent = {
		Impl_->TargetWidth, Impl_->TargetHeight};
	rendering.layerCount = 1;
	rendering.colorAttachmentCount = 1;
	rendering.pColorAttachments = &color;
	vkCmdBeginRendering(InCommandBuffer, &rendering);

	const VkViewport viewport{
		0.0F, static_cast<OaF32>(Impl_->TargetHeight),
		static_cast<OaF32>(Impl_->TargetWidth),
		-static_cast<OaF32>(Impl_->TargetHeight),
		0.0F, 1.0F};
	const VkRect2D scissor{
		{0, 0}, {Impl_->TargetWidth, Impl_->TargetHeight}};
	vkCmdSetViewport(InCommandBuffer, 0, 1, &viewport);
	vkCmdSetScissor(InCommandBuffer, 0, 1, &scissor);
	vkCmdBindPipeline(
		InCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		Impl_->Pipeline);
	const VkBuffer vertex =
		static_cast<VkBuffer>(Impl_->VertexBuffer.Buffer);
	const VkDeviceSize vertexOffset = 0;
	vkCmdBindVertexBuffers(
		InCommandBuffer, 0, 1, &vertex, &vertexOffset);
	vkCmdBindIndexBuffer(
		InCommandBuffer,
		static_cast<VkBuffer>(Impl_->IndexBuffer.Buffer),
		0, VK_INDEX_TYPE_UINT32);

	for (const auto& draw : Impl_->Draws) {
		VkDescriptorSet set = descriptorSet(draw);
		if (!set) continue;
		vkCmdBindDescriptorSets(
			InCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			Impl_->PipelineLayout, 0, 1, &set, 0, nullptr);
		CanvasPushConstants push{
			Vlm::MatrixMul(
				draw.Model, draw.Camera.GetViewProjectionMatrix()),
			draw.Tint};
		vkCmdPushConstants(
			InCommandBuffer, Impl_->PipelineLayout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(push), &push);
		vkCmdDrawIndexed(InCommandBuffer, 6, 1, 0, 0, 0);
	}
	if (Impl_->DetectionPipeline && !Impl_->DetectionDraws.Empty()) {
		const VkDescriptorSet bindlessSet =
			static_cast<VkDescriptorSet>(
				Impl_->Engine->Bindless.DescriptorSet);
		vkCmdBindPipeline(
			InCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			Impl_->DetectionPipeline);
		vkCmdBindDescriptorSets(
			InCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			Impl_->DetectionPipelineLayout, 0, 1,
			&bindlessSet, 0, nullptr);
		for (const auto& draw : Impl_->DetectionDraws) {
			DetectionPushConstants push{
				Vlm::MatrixMul(
					draw.Model,
					draw.Camera.GetViewProjectionMatrix()),
				draw.Color,
				draw.InstanceBufferIndex,
				Impl_->TargetWidth,
				Impl_->TargetHeight,
				draw.ThicknessPixels};
			vkCmdPushConstants(
				InCommandBuffer, Impl_->DetectionPipelineLayout,
				VK_SHADER_STAGE_VERTEX_BIT |
					VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(push), &push);
			vkCmdDraw(
				InCommandBuffer, 24,
				draw.InstanceCount, 0, 0);
		}
	}
	if (Impl_->GlyphPipeline && !Impl_->GlyphDraws.Empty()) {
		const VkDescriptorSet bindlessSet =
			static_cast<VkDescriptorSet>(
				Impl_->Engine->Bindless.DescriptorSet);
		vkCmdBindPipeline(
			InCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			Impl_->GlyphPipeline);
		vkCmdBindDescriptorSets(
			InCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			Impl_->GlyphPipelineLayout, 0, 1,
			&bindlessSet, 0, nullptr);
		for (const auto& draw : Impl_->GlyphDraws) {
			GlyphPushConstants push{
				Vlm::MatrixMul(
					draw.Model,
					draw.Camera.GetViewProjectionMatrix()),
				draw.InstanceBufferIndex,
				draw.AtlasBufferIndex,
				draw.ReferenceWidth,
				draw.ReferenceHeight,
				draw.AtlasWidth,
				draw.AtlasHeight,
				draw.AtlasPxRange,
				0};
			vkCmdPushConstants(
				InCommandBuffer, Impl_->GlyphPipelineLayout,
				VK_SHADER_STAGE_VERTEX_BIT |
					VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(push), &push);
			vkCmdDraw(
				InCommandBuffer, 6,
				draw.InstanceCount, 0, 0);
		}
	}
	vkCmdEndRendering(InCommandBuffer);

	VkImageMemoryBarrier toTransfer{};
	toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	toTransfer.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	toTransfer.image = Impl_->TargetImage;
	toTransfer.subresourceRange = range;
	toTransfer.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

	VkImageMemoryBarrier destinationReady{};
	destinationReady.sType =
		VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	destinationReady.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	destinationReady.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	destinationReady.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	destinationReady.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	destinationReady.image = InDestinationImage;
	destinationReady.subresourceRange = range;
	destinationReady.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	destinationReady.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	VkImageMemoryBarrier transferBarriers[2] = {
		toTransfer, destinationReady};
	vkCmdPipelineBarrier(
		InCommandBuffer,
		VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
			VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		0, 0, nullptr, 0, nullptr, 2, transferBarriers);

	VkImageCopy copy{};
	copy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.srcSubresource.layerCount = 1;
	copy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copy.dstSubresource.layerCount = 1;
	copy.extent = {Impl_->TargetWidth, Impl_->TargetHeight, 1};
	vkCmdCopyImage(
		InCommandBuffer,
		Impl_->TargetImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		InDestinationImage, VK_IMAGE_LAYOUT_GENERAL,
		1, &copy);

	VkImageMemoryBarrier toCompute = destinationReady;
	toCompute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	toCompute.dstAccessMask =
		VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	vkCmdPipelineBarrier(
		InCommandBuffer,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0, 0, nullptr, 0, nullptr, 1, &toCompute);
	Impl_->TargetLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
}
