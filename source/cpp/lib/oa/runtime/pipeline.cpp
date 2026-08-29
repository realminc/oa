#include <oa/runtime/pipeline.h>
#include "descriptorValidation.h"
#include <oa/runtime/device.h>
#include <oa/runtime/allocator.h>
#include <vkl/vkl.h>
#include <oa/core/filesystem.h>
#include <oa/core/log.h>
#include <oa/core/std/chrono.h>
#include "../core/logAccess.h"
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/envFlag.h>
#include <oa/core/thread.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif


namespace {
constexpr const char* pipelineCacheFile = "pipeline.vcache";

// SIMD-width experiment. OA_GEMM_SUBGROUP_SIZE=8|16|32 pins the compute subgroup
// size on every compute pipeline via VkPipelineShaderStageRequiredSubgroupSize-
// CreateInfo — used to align to the Intel Xe native fp32 vector width (SIMD16),
// which the Intel compiler otherwise often lowers to SIMD8. Holds the struct;
// the caller keeps it alive through vkCreateComputePipelines. Unset → no-op.
struct ForcedSubgroup {
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo info{};
	bool active = false;
};

inline void maybeForceSubgroupSize(
	VkPipelineShaderStageCreateInfo& inOutStage,
	ForcedSubgroup& outHolder
) {
	const oa::I64 sz = oa::EnvFlag::getInt("OA_GEMM_SUBGROUP_SIZE", 0);
	if (sz == 8 || sz == 16 || sz == 32) {
		outHolder.info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
		outHolder.info.requiredSubgroupSize = static_cast<uint32_t>(sz);
		outHolder.info.pNext = nullptr;
		inOutStage.pNext = &outHolder.info;   // stage pNext is null on the compute paths here
		outHolder.active = true;
	}
}
} // namespace


oa::Result<oa::PipelineCache> oa::PipelineCache::create(const oavk::Device& inDevice, const oa::String& inCacheDir) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	oa::Vector<oa::U8> cacheData;
	if (!inCacheDir.empty()) {
		oa::Path cachePath = oa::Path(inCacheDir) / pipelineCacheFile;
		auto loaded = oa::Filesystem::readBinary(cachePath);
		if (loaded.isOk()) {
			cacheData = oa::move(loaded.getValue());
			OaLogInfo(oa::LogComponent::Compute, "pipeline cache: loaded {} bytes from {}",	cacheData.size(), cachePath.string().cStr());
		}
	}

	VkPipelineCacheCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	ci.initialDataSize = cacheData.size();
	ci.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

	VkPipelineCache cache = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreatePipelineCache(dev, &ci, nullptr, &cache);
	if (r != VK_SUCCESS) {
		OaLogWarn(oa::LogComponent::Compute, "pipeline cache: creation failed, continuing without cache");
		oa::PipelineCache pc;
		return pc;
	}

	oa::PipelineCache pc;
	pc.cache = cache;
	pc.initialDataBytes = static_cast<oa::U64>(cacheData.size());
	return pc;
}


void oa::PipelineCache::save(const oavk::Device& inDevice, const oa::String& inCacheDir) const {
	if (!cache || inCacheDir.empty()) {
		return;
	}

	VkDevice dev = static_cast<VkDevice>(inDevice.device);
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(cache);

	size_t dataSize = 0;
	(void)inDevice.deviceDispatch.vkGetPipelineCacheData(dev, vkCache, &dataSize, nullptr);
	if (dataSize == 0) {
		return;
	}

	oa::Vector<oa::U8> data(dataSize);
	(void)inDevice.deviceDispatch.vkGetPipelineCacheData(dev, vkCache, &dataSize, data.data());

	oa::Path cacheDir(inCacheDir);
	(void)oa::Filesystem::createDirectories(cacheDir);
	oa::Path cachePath = cacheDir / pipelineCacheFile;
	(void)oa::Filesystem::writeBinary(cachePath, oa::Span<const oa::U8>(data.data(), data.size()));
	OaLogInfo(oa::LogComponent::Compute, "pipeline cache: saved {} bytes to {}",
		dataSize, cachePath.string().cStr());
}

void oa::PipelineCache::destroy(const oavk::Device& inDevice) {
	if (cache) {
		VkDevice dev = static_cast<VkDevice>(inDevice.device);
		inDevice.deviceDispatch.vkDestroyPipelineCache(dev, static_cast<VkPipelineCache>(cache), nullptr);
		cache = nullptr;
	}
	initialDataBytes = 0;
}

oa::Result<oa::ComputePipeline> oa::ComputePipeline::create(
	const oavk::Device& inDevice,
	oa::Span<const oa::U8> inSpirv,
	const oa::PipelineSpec& inSpec,
	void* inPipelineCache,
	void* inBindlessPipelineLayout)
{
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkShaderModuleCreateInfo smCI{};
	smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	smCI.codeSize = inSpirv.size();
	smCI.pCode = reinterpret_cast<const oa::U32*>(inSpirv.data());

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateShaderModule(dev, &smCI, nullptr, &shaderModule);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::ShaderCompileError, "vkCreateShaderModule failed");
	}

	VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	bool ownsLayout = false;

	if (inBindlessPipelineLayout) {
		// bindless: shared pipeline layout, no per-pipeline descriptor layout
		pipelineLayout = static_cast<VkPipelineLayout>(inBindlessPipelineLayout);
	} else {
		ownsLayout = true;

		oa::Vector<VkDescriptorSetLayoutBinding> bindings(inSpec.numBindings);
		for (oa::U32 i = 0; i < inSpec.numBindings; i++) {
			bindings[i] = {};
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo dslCI{};
		dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslCI.bindingCount = inSpec.numBindings;
		dslCI.pBindings = bindings.data();

		r = inDevice.deviceDispatch.vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);
		if (r != VK_SUCCESS) {
			inDevice.deviceDispatch.vkDestroyShaderModule(dev, shaderModule, nullptr);
			return oa::Status::error(oa::StatusCode::PipelineError, "vkCreateDescriptorSetLayout failed");
		}

		VkPushConstantRange pushRange = {
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = inSpec.pushConstantBytes
		};

		VkPipelineLayoutCreateInfo plCI{};
		plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		plCI.setLayoutCount = 1;
		plCI.pSetLayouts = &dsl;
		plCI.pushConstantRangeCount = (inSpec.pushConstantBytes > 0) ? 1 : 0;
		plCI.pPushConstantRanges = (inSpec.pushConstantBytes > 0) ? &pushRange : nullptr;

		r = inDevice.deviceDispatch.vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
		if (r != VK_SUCCESS) {
			inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
			inDevice.deviceDispatch.vkDestroyShaderModule(dev, shaderModule, nullptr);
			return oa::Status::error(oa::StatusCode::PipelineError, "vkCreatePipelineLayout failed");
		}
	}

	oa::Vector<VkSpecializationMapEntry> specEntries;
	oa::Vector<oa::U32> specData;
	VkSpecializationInfo specInfo{};

	if (!inSpec.specConstants.empty()) {
		specEntries.resize(inSpec.specConstants.size());
		specData.resize(inSpec.specConstants.size());
		for (oa::U32 i = 0; i < inSpec.specConstants.size(); i++) {
			specEntries[i].constantID = inSpec.specConstants[i].id;
			specEntries[i].offset = i * sizeof(oa::U32);
			specEntries[i].size = sizeof(oa::U32);
			specData[i] = inSpec.specConstants[i].value;
		}
		specInfo.mapEntryCount = static_cast<oa::U32>(specEntries.size());
		specInfo.pMapEntries = specEntries.data();
		specInfo.dataSize = specData.size() * sizeof(oa::U32);
		specInfo.pData = specData.data();
	}

	VkComputePipelineCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpCI.stage.module = shaderModule;
	cpCI.stage.pName = "main";
	cpCI.stage.pSpecializationInfo = inSpec.specConstants.empty() ? nullptr : &specInfo;
	cpCI.layout = pipelineLayout;

	VkPipeline linkedPipeline = VK_NULL_HANDLE;
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(inPipelineCache);
	ForcedSubgroup forcedSg;
	maybeForceSubgroupSize(cpCI.stage, forcedSg);
	r = inDevice.deviceDispatch.vkCreateComputePipelines(
		dev, vkCache, 1, &cpCI, nullptr, &linkedPipeline);
	if (r != VK_SUCCESS && forcedSg.active) {
		// Forced subgroup size unsupported for this shader/device — retry
		// unconstrained so the experiment knob can't brick pipeline creation.
		cpCI.stage.pNext = nullptr;
		r = inDevice.deviceDispatch.vkCreateComputePipelines(
			dev, vkCache, 1, &cpCI, nullptr, &linkedPipeline);
	}
	inDevice.deviceDispatch.vkDestroyShaderModule(dev, shaderModule, nullptr);
	if (r != VK_SUCCESS) {
		// bindless path uses engine-owned layout — must not destroy (see destroy()).
		if (ownsLayout) {
			inDevice.deviceDispatch.vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
			inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		}
		return oa::Status::error(oa::StatusCode::PipelineError, "vkCreateComputePipelines failed");
	}

	oa::ComputePipeline p;
	p.pipeline = linkedPipeline;
	p.pipelineLayout = pipelineLayout;
	p.descriptorSetLayout = ownsLayout ? dsl : nullptr;
	p.descriptorPool = nullptr;
	p.descriptorSet = nullptr;
	p.numBindings = inSpec.numBindings;
	p.bindless = (inBindlessPipelineLayout != nullptr);
	
	// Extract native DTYPE from spec constants (ID=0 is DTYPE by convention)
	p.nativeDtype = 0;  // Default to FP32
	for (const auto& sc : inSpec.specConstants) {
		if (sc.id == 0) {
			p.nativeDtype = sc.value;
			break;
		}
	}
	
	return p;
}

void oa::ComputePipeline::destroy(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);
	if (descriptorPool) inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(descriptorPool), nullptr);
	if (pipeline) inDevice.deviceDispatch.vkDestroyPipeline(dev, static_cast<VkPipeline>(pipeline), nullptr);
	if (!bindless) {
		if (pipelineLayout) inDevice.deviceDispatch.vkDestroyPipelineLayout(dev, static_cast<VkPipelineLayout>(pipelineLayout), nullptr);
		if (descriptorSetLayout) inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, static_cast<VkDescriptorSetLayout>(descriptorSetLayout), nullptr);
	}
	pipeline = pipelineLayout = descriptorSetLayout = descriptorPool = descriptorSet = nullptr;
	bindless = false;
}

oa::Status oa::ComputePipeline::allocDescriptorSet(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	if (descriptorPool) {
		inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(descriptorPool), nullptr);
		descriptorPool = nullptr;
		descriptorSet = nullptr;
	}

	oa::U32 numDesc = numBindings > 0 ? numBindings : 16;

	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = numDesc;

	VkDescriptorPoolCreateInfo dpCI{};
	dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpCI.maxSets = 1;
	dpCI.poolSizeCount = 1;
	dpCI.pPoolSizes = &poolSize;

	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateDescriptorPool(dev, &dpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::PipelineError, "vkCreateDescriptorPool failed");
	}

	VkDescriptorSetLayout dsl = static_cast<VkDescriptorSetLayout>(descriptorSetLayout);
	VkDescriptorSetAllocateInfo dsAI{};
	dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAI.descriptorPool = pool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts = &dsl;

	VkDescriptorSet ds = VK_NULL_HANDLE;
	r = inDevice.deviceDispatch.vkAllocateDescriptorSets(dev, &dsAI, &ds);
	if (r != VK_SUCCESS) {
		inDevice.deviceDispatch.vkDestroyDescriptorPool(dev, pool, nullptr);
		return oa::Status::error(oa::StatusCode::PipelineError, "vkAllocateDescriptorSets failed");
	}

	descriptorPool = pool;
	descriptorSet = ds;
	return oa::Status::ok();
}

oa::Status oa::ComputePipeline::writeStorageBuffer(
	const oavk::Device& inDevice,
	oa::U32 inBinding,
	const oavk::Buffer& inBuffer)
{
	OA_RETURN_IF_ERROR(
		oavk::validateStorageBufferDescriptor(inDevice, inBuffer));
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkDescriptorBufferInfo bufInfo{};
	bufInfo.buffer = static_cast<::VkBuffer>(inBuffer.buffer);
	bufInfo.offset = 0;
	bufInfo.range = inBuffer.descriptorRange();

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = static_cast<VkDescriptorSet>(descriptorSet);
	write.dstBinding = inBinding;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufInfo;

	inDevice.deviceDispatch.vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
	return oa::Status::ok();
}

// ─── pipeline Library ──────────────────────────────────────────────────────

oa::Result<oa::PipelineLibrary> oa::PipelineLibrary::create(
	const oavk::Device& inDevice,
	oa::Span<const oa::U8> inSpirv,
	const oa::PipelineSpec& inSpec,
	void* inPipelineCache,
	void* inBindlessPipelineLayout)
{
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkShaderModuleCreateInfo smCI{};
	smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	smCI.codeSize = inSpirv.size();
	smCI.pCode = reinterpret_cast<const oa::U32*>(inSpirv.data());

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	VkResult r = inDevice.deviceDispatch.vkCreateShaderModule(dev, &smCI, nullptr, &shaderModule);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::ShaderCompileError,
			"pipeline library: vkCreateShaderModule failed");
	}

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	if (inBindlessPipelineLayout) {
		pipelineLayout = static_cast<VkPipelineLayout>(inBindlessPipelineLayout);
	} else {
		oa::Vector<VkDescriptorSetLayoutBinding> bindings(inSpec.numBindings);
		for (oa::U32 i = 0; i < inSpec.numBindings; i++) {
			bindings[i] = {};
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo dslCI{};
		dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslCI.bindingCount = inSpec.numBindings;
		dslCI.pBindings = bindings.data();

		VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
		r = inDevice.deviceDispatch.vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);
		if (r != VK_SUCCESS) {
			inDevice.deviceDispatch.vkDestroyShaderModule(dev, shaderModule, nullptr);
			return oa::Status::error(oa::StatusCode::PipelineError,
				"pipeline library: vkCreateDescriptorSetLayout failed");
		}

		VkPushConstantRange pushRange = {
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = inSpec.pushConstantBytes
		};

		VkPipelineLayoutCreateInfo plCI{};
		plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		plCI.setLayoutCount = 1;
		plCI.pSetLayouts = &dsl;
		plCI.pushConstantRangeCount = (inSpec.pushConstantBytes > 0) ? 1 : 0;
		plCI.pPushConstantRanges = (inSpec.pushConstantBytes > 0) ? &pushRange : nullptr;

		r = inDevice.deviceDispatch.vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
		inDevice.deviceDispatch.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		if (r != VK_SUCCESS) {
			inDevice.deviceDispatch.vkDestroyShaderModule(dev, shaderModule, nullptr);
			return oa::Status::error(oa::StatusCode::PipelineError,
				"pipeline library: vkCreatePipelineLayout failed");
		}
	}

	VkComputePipelineCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpCI.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
	cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpCI.stage.module = shaderModule;
	cpCI.stage.pName = "main";
	cpCI.layout = pipelineLayout;

	VkPipeline libraryPipeline = VK_NULL_HANDLE;
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(inPipelineCache);
	ForcedSubgroup forcedSg;
	maybeForceSubgroupSize(cpCI.stage, forcedSg);
	r = inDevice.deviceDispatch.vkCreateComputePipelines(
		dev, vkCache, 1, &cpCI, nullptr, &libraryPipeline);
	if (r != VK_SUCCESS && forcedSg.active) {
		cpCI.stage.pNext = nullptr;   // retry unconstrained (see oa::ComputePipeline::create)
		r = inDevice.deviceDispatch.vkCreateComputePipelines(
			dev, vkCache, 1, &cpCI, nullptr, &libraryPipeline);
	}
	inDevice.deviceDispatch.vkDestroyShaderModule(dev, shaderModule, nullptr);
	if (r != VK_SUCCESS) {
		if (!inBindlessPipelineLayout) {
			inDevice.deviceDispatch.vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
		}
		return oa::Status::error(oa::StatusCode::PipelineError,
			"pipeline library: vkCreateComputePipelines (LIBRARY_BIT) failed");
	}

	oa::PipelineLibrary lib;
	lib.pipeline = libraryPipeline;
	lib.pipelineLayout = pipelineLayout;
	return lib;
}

void oa::PipelineLibrary::destroy(const oavk::Device& inDevice) {
	VkDevice dev = static_cast<VkDevice>(inDevice.device);
	if (pipeline) {
		inDevice.deviceDispatch.vkDestroyPipeline(dev, static_cast<VkPipeline>(pipeline), nullptr);
		pipeline = nullptr;
	}
}

oa::Result<oa::ComputePipeline> oa::PipelineLibrary::link(
	const oavk::Device& inDevice,
	void* inPipelineCache,
	void* inBindlessPipelineLayout) const
{
	VkDevice dev = static_cast<VkDevice>(inDevice.device);

	VkPipeline lib = static_cast<VkPipeline>(pipeline);
	VkPipelineLibraryCreateInfoKHR libInfo{};
	libInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
	libInfo.libraryCount = 1;
	libInfo.pLibraries = &lib;

	VkComputePipelineCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpCI.pNext = &libInfo;
	cpCI.layout = static_cast<VkPipelineLayout>(pipelineLayout);

	VkPipeline linkedPipeline = VK_NULL_HANDLE;
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(inPipelineCache);
	VkResult r = inDevice.deviceDispatch.vkCreateComputePipelines(
		dev, vkCache, 1, &cpCI, nullptr, &linkedPipeline);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::PipelineError,
			"pipeline library: link failed");
	}

	oa::ComputePipeline p;
	p.pipeline = linkedPipeline;
	p.pipelineLayout = static_cast<VkPipelineLayout>(pipelineLayout);
	p.bindless = (inBindlessPipelineLayout != nullptr);
	return p;
}

// ─── pipeline registry ────────────────────────────────────────────────────

namespace {
oa::String makePipelineKey(oa::StringView inName, const oa::PipelineSpec& inSpec) {
	oa::String key(inName);
	for (const auto& sc : inSpec.specConstants) {
		key += "|";
		key += oa::toString(sc.id);
		key += "=";
		key += oa::toString(sc.value);
	}
	return key;
}
} // namespace

oa::Status oa::PipelineRegistry::init(
	const oavk::Device& inDevice, const oa::String& inCacheDir,
	void* inBindlessPipelineLayout)
{
	device_ = &inDevice;
	cacheDir_ = inCacheDir;
	bindlessPipelineLayout_ = inBindlessPipelineLayout;
	auto cacheResult = oa::PipelineCache::create(inDevice, inCacheDir);
	if (cacheResult.isOk()) cache_ = cacheResult.getValue();
	return oa::Status::ok();
}

void oa::PipelineRegistry::destroy(const oavk::Device& inDevice) {
	cache_.save(inDevice, cacheDir_);
	cache_.destroy(inDevice);
	oa::UniqueLock lock(*mutex_);
	for (auto& [name, pipe] : registry_) {
		pipe.destroy(inDevice);
	}
	registry_.clear();
	device_ = nullptr;  // clear dangling pointer to prevent use-after-free
}

oa::Status oa::PipelineRegistry::ensurePipeline(
	const oavk::Device& inDevice,
	oa::StringView inName,
	oa::Span<const oa::U8> inSpirv,
	const oa::PipelineSpec& inSpec)
{
	oa::String key = makePipelineKey(inName, inSpec);

	{
		oa::SharedLock lock(*mutex_);
		if (registry_.contains(key)) return oa::Status::ok();
	}

	// Direct creation + pipeline cache. oa::PipelineLibrary::create/link is
	// available for manual use but not auto-integrated here -- pipeline
	// libraries target multi-stage (ray tracing / graphics) pipelines;
	// for single-stage compute the VkPipelineCache already provides
	// equivalent cold-start reduction.
	auto result = oa::ComputePipeline::create(
		inDevice, inSpirv, inSpec, cache_.cache, bindlessPipelineLayout_);
	if (!result.isOk() && cache_.cache) {
		OaLogWarn(oa::LogComponent::Compute,
			"pipeline creation failed with cache for {}, dropping {} and retrying uncached",
			oa::String(inName).cStr(), pipelineCacheFile);
		cache_.destroy(inDevice);
		if (!cacheDir_.empty()) {
			(void)oa::Filesystem::removeFile(oa::Path(cacheDir_) / pipelineCacheFile);
		}
		result = oa::ComputePipeline::create(
			inDevice, inSpirv, inSpec, nullptr, bindlessPipelineLayout_);
	}
	if (!result.isOk()) return result.getStatus();

	oa::UniqueLock lock(*mutex_);
	if (registry_.contains(key)) {
		result.getValue().destroy(inDevice);
		return oa::Status::ok();
	}
	registry_.emplace(oa::move(key), oa::move(result.getValue()));
	return oa::Status::ok();
}

oa::Status oa::PipelineRegistry::ensurePipelinesParallel(
	const oavk::Device& inDevice,
	oa::Span<const oa::PipelineLoadRequest> inRequests,
	oa::U32 inWorkerCount,
	oa::Vector<oa::Status>* outStatuses)
{
	const oa::U32 requestCount = static_cast<oa::U32>(inRequests.size());
	if (outStatuses) {
		outStatuses->resize(requestCount);
	}
	if (requestCount == 0) return oa::Status::ok();

	const oa::U32 workerCount = oa::max<oa::U32>(1u,
		oa::min<oa::U32>(inWorkerCount, requestCount));
	if (workerCount == 1) {
		oa::Status firstError = oa::Status::ok();
		for (oa::U32 i = 0; i < requestCount; ++i) {
			const auto& request = inRequests[i];
			oa::Status status = ensurePipeline(
				inDevice, request.name, request.spirv, request.spec);
			if (outStatuses) (*outStatuses)[i] = status;
			if (firstError.isOk() && status.isError()) firstError = status;
		}
		return firstError;
	}

	class ParallelBuildResult {
	public:
		oa::Status status;
		oa::ComputePipeline pipeline;
		bool needsBuild = true;
		bool hasPipeline = false;
	};

	oa::Vector<ParallelBuildResult> results(requestCount);
	{
		oa::SharedLock lock(*mutex_);
		for (oa::U32 i = 0; i < requestCount; ++i) {
			if (registry_.contains(makePipelineKey(inRequests[i].name, inRequests[i].spec))) {
				results[i].needsBuild = false;
			}
		}
	}

	VkDevice device = static_cast<VkDevice>(inDevice.device);
	VkPipelineCache primaryCache = static_cast<VkPipelineCache>(cache_.cache);

	// snapshot the primary cache once, before workers start. Feeding this same
	// immutable snapshot to each worker preserves warm-cache startup while keeping
	// every vkCreateComputePipelines call on a separately synchronized cache.
	oa::Vector<oa::U8> initialData;
	if (primaryCache != VK_NULL_HANDLE) {
		size_t size = 0;
		if (inDevice.deviceDispatch.vkGetPipelineCacheData(device, primaryCache, &size, nullptr) == VK_SUCCESS && size != 0) {
			initialData.resize(size);
			if (inDevice.deviceDispatch.vkGetPipelineCacheData(device, primaryCache, &size, initialData.data()) != VK_SUCCESS) {
				initialData.clear();
			} else if (size < initialData.size()) {
				initialData.resize(size);
			}
		}
	}

	oa::Vector<VkPipelineCache> workerCaches(workerCount);
	for (oa::U32 i = 0; i < workerCount; ++i) {
		VkPipelineCacheCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		ci.initialDataSize = initialData.size();
		ci.pInitialData = initialData.empty() ? nullptr : initialData.data();
		if (inDevice.deviceDispatch.vkCreatePipelineCache(device, &ci, nullptr, &workerCaches[i]) != VK_SUCCESS) {
			workerCaches[i] = VK_NULL_HANDLE;
			OaLogWarn(oa::LogComponent::Compute,
				"shader preload worker {}: pipeline-cache creation failed; compiling uncached", i);
		}
	}

	oa::Atomic<oa::U32> nextIndex{0};
	const oa::LogSelection logSelection = oa::LogAccess::currentSelection();
	oa::Vector<oa::Thread> workers;
	workers.reserve(workerCount);
	oa::Status launchStatus = oa::Status::ok();
	for (oa::U32 worker = 0; worker < workerCount; ++worker) {
		auto created = oa::Thread::create([&, worker] {
			oa::LogAccess::Scope logScope(logSelection);
			for (;;) {
				const oa::U32 index = nextIndex.fetchAdd(
					1, oa::MemoryOrder::Relaxed);
				if (index >= requestCount) break;
				auto& build = results[index];
				if (!build.needsBuild) continue;

				const auto& request = inRequests[index];
				auto result = oa::ComputePipeline::create(
					inDevice, request.spirv, request.spec,
					workerCaches[worker], bindlessPipelineLayout_);
				if (!result.isOk() && workerCaches[worker] != VK_NULL_HANDLE) {
					// A corrupt/incompatible cache entry must not make the shader unloadable.
					// Retry this pipeline without mutating the shared primary cache.
					result = oa::ComputePipeline::create(
						inDevice, request.spirv, request.spec,
						nullptr, bindlessPipelineLayout_);
				}
				if (!result.isOk()) {
					build.status = result.getStatus();
					continue;
				}
				build.pipeline = oa::move(result.getValue());
				build.hasPipeline = true;
			}
		});
		if (created.isError()) {
			launchStatus = created.getStatus();
			break;
		}
		workers.pushBack(oa::move(*created));
	}
	for (auto& worker : workers) {
		const oa::Status joinStatus = worker.join();
		if (launchStatus.isOk() and joinStatus.isError()) launchStatus = joinStatus;
	}
	if (launchStatus.isError()) {
		for (VkPipelineCache workerCache : workerCaches) {
			if (workerCache != VK_NULL_HANDLE) {
				inDevice.deviceDispatch.vkDestroyPipelineCache(
					device, workerCache, nullptr);
			}
		}
		return launchStatus;
	}

	// Merge only after all workers stop touching their caches. This satisfies the
	// vulkan external-synchronization requirements for both source and destination.
	if (primaryCache != VK_NULL_HANDLE) {
		oa::Vector<VkPipelineCache> mergeCaches;
		mergeCaches.reserve(workerCount);
		for (VkPipelineCache cache : workerCaches) {
			if (cache != VK_NULL_HANDLE) mergeCaches.pushBack(cache);
		}
		if (!mergeCaches.empty()) {
			const VkResult merged = inDevice.deviceDispatch.vkMergePipelineCaches(
				device, primaryCache, static_cast<oa::U32>(mergeCaches.size()), mergeCaches.data());
			if (merged != VK_SUCCESS) {
				OaLogWarn(oa::LogComponent::Compute,
					"shader preload: vkMergePipelineCaches failed (VkResult={})",
					static_cast<int>(merged));
			}
		}
	}
	for (VkPipelineCache cache : workerCaches) {
		if (cache != VK_NULL_HANDLE) inDevice.deviceDispatch.vkDestroyPipelineCache(device, cache, nullptr);
	}

	oa::Status firstError = oa::Status::ok();
	{
		oa::UniqueLock lock(*mutex_);
		for (oa::U32 i = 0; i < requestCount; ++i) {
			auto& build = results[i];
			if (build.hasPipeline) {
				oa::String key = makePipelineKey(inRequests[i].name, inRequests[i].spec);
				if (registry_.contains(key)) {
					build.pipeline.destroy(inDevice);
				} else {
					registry_.emplace(oa::move(key), oa::move(build.pipeline));
				}
			}
			if (outStatuses) (*outStatuses)[i] = build.status;
			if (firstError.isOk() && build.status.isError()) firstError = build.status;
		}
	}
	return firstError;
}

oa::Status oa::PipelineRegistry::ensurePipelinesOnDemand(
	oa::Span<const oa::PipelineVariantRequest> inRequests)
{
	if (device_ == nullptr or inRequests.empty()) return oa::Status::ok();

	oa::Vector<oa::PipelineLoadRequest> requests;
	oa::HashMap<oa::String, oa::Bool> planned;
	requests.reserve(inRequests.size());
	for (const auto& request : inRequests) {
		if (request.dtype > 1U) {
			return oa::Status::invalidArgument(
				"on-demand pipeline request has invalid storage DTYPE");
		}
		if (request.name.empty()
			or not oa::computeKernelUsesDefaultBindlessPipeline(request.name.cStr())) {
			continue;
		}

		oa::PipelineSpec spec;
		spec.numBindings = 16;
		spec.pushConstantBytes = 128;
		spec.specConstants = oa::Vector<oa::SpecConstant>{
			oa::SpecConstant{.id = 0, .value = request.dtype}};
		oa::String key = makePipelineKey(request.name, spec);
		{
			oa::SharedLock lock(*mutex_);
			const auto exact = registry_.find(key);
			if (exact != registry_.end()) continue;
			const auto bare = registry_.find(request.name);
			if (bare != registry_.end() and bare->second.nativeDtype == request.dtype) {
				continue;
			}
		}
		if (planned.contains(key)) continue;

		const auto* spirv = oavk::findSpirv(request.name.cStr());
		if (spirv == nullptr) {
			OaLogWarn(oa::LogComponent::Compute,
				"SPIR-V not found for on-demand shader '{}'",
				request.name.cStr());
			return oa::Status::notFound("SPIR-V not found in registry");
		}
		planned.emplace(oa::move(key), true);
		requests.pushBack({
			.name = request.name,
			.spirv = oa::Span<const oa::U8>(spirv->data, spirv->size),
			.spec = oa::move(spec),
		});
	}
	if (requests.empty()) return oa::Status::ok();

	const oa::I64 configuredThreads = oa::EnvFlag::getInt("OA_SHADER_LOAD_THREADS", 0);
	const oa::U32 hardwareThreads = oa::Thread::hardwareConcurrency();
	oa::U32 loadThreads = 1;
	if (configuredThreads > 0) {
		loadThreads = static_cast<oa::U32>(oa::min<oa::I64>(configuredThreads, 64));
	} else if (not hasInitialCacheData()) {
		loadThreads = oa::min<oa::U32>(
			oa::max<oa::U32>(1u, hardwareThreads / 2u), 8u);
	}
	loadThreads = oa::max<oa::U32>(1u,
		oa::min<oa::U32>(loadThreads, static_cast<oa::U32>(requests.size())));

	OaLogInfo(oa::LogComponent::Compute,
		"Loading {} shader pipeline{} on demand ({} thread{}, {} cache)",
		requests.size(), requests.size() == 1 ? "" : "s",
		loadThreads, loadThreads == 1 ? "" : "s",
		hasInitialCacheData() ? "warm" : "cold");

	const auto loadBegin = oa::steadyNow();
	oa::Vector<oa::Status> statuses;
	const oa::Status loadStatus = ensurePipelinesParallel(
		*device_,
		oa::Span<const oa::PipelineLoadRequest>(requests.data(), requests.size()),
		loadThreads,
		&statuses);
	const oa::F64 loadMs = (oa::steadyNow() - loadBegin).toMilliseconds();

	oa::U32 loaded = 0;
	oa::U32 failed = 0;
	for (oa::Usize index = 0; index < statuses.size(); ++index) {
		if (statuses[index].isOk()) {
			++loaded;
		} else {
			++failed;
			OaLogWarn(oa::LogComponent::Compute,
				"Failed to load shader '{}' on demand: {}",
				requests[index].name.cStr(), statuses[index].getMessage().cStr());
		}
	}
	OaLogInfo(oa::LogComponent::Compute,
		"Loaded {}/{} shader pipeline{} on demand (failed={}, threads={}, {:.2f} ms)",
		loaded, requests.size(), requests.size() == 1 ? "" : "s",
		failed, loadThreads, loadMs);
	return loadStatus;
}

oa::Status oa::PipelineRegistry::tryLoadOnDemand(
	const oavk::Device& inDevice,
	oa::StringView inName,
	oa::U32 inDtype)
{
	oa::String kernelName(inName);
	if (!oa::computeKernelUsesDefaultBindlessPipeline(kernelName.cStr())) {
		return oa::Status::notFound("kernel uses a non-default image/presentation pipeline layout");
	}

	// Look up SPIR-V
	auto* spirv = oavk::findSpirv(kernelName.cStr());
	if (!spirv) {
		OaLogWarn(oa::LogComponent::Compute, "TryLoadOnDemand: SPIR-V not found for '{}'", kernelName.cStr());
		return oa::Status::notFound("SPIR-V not found in registry");
	}
	
	#ifdef __ANDROID__
	__android_log_print(ANDROID_LOG_INFO, "OA", "Loading pipeline %s dtype=%u (%u bytes)",
		kernelName.cStr(), inDtype, spirv->size);
	#endif
	
	// Match the embedded preload ABI exactly. All default bindless shaders share
	// the 16-slot heap layout and 128-byte push range; DTYPE is specialization
	// constant 0. Keeping lazy and eager creation identical is essential because
	// graph nodes resolve the variant by their operand dtype.
	oa::PipelineSpec spec;
	spec.numBindings = 16;
	spec.pushConstantBytes = 128;
	spec.specConstants = oa::Vector<oa::SpecConstant>{
		oa::SpecConstant{.id = 0, .value = inDtype}};

	// load it
	oa::Status status = ensurePipeline(inDevice, kernelName,
		oa::Span<const oa::U8>(spirv->data, spirv->size), spec);
	
	if (status.isOk()) {
		#ifdef __ANDROID__
		__android_log_print(ANDROID_LOG_INFO, "OA", "Loaded pipeline %s dtype=%u",
			kernelName.cStr(), inDtype);
		#endif
	} else {
		OaLogWarn(oa::LogComponent::Compute, "TryLoadOnDemand: Failed to load '{}'",
			kernelName.cStr());
	}
	
	return status;
}

oa::ComputePipeline& oa::PipelineRegistry::getPipeline(oa::StringView inName, oa::U32 inDtype) {
	// Single static null pipeline - must be outside all scopes
	static oa::ComputePipeline sNull;
	if (inDtype > 1U) {
		OaLogError(oa::LogComponent::Compute,
			"pipeline lookup rejected invalid storage DTYPE={} for '{}'",
			inDtype, oa::String(inName).cStr());
		return sNull;
	}
	
	oa::String key(inName);
	key += "|0=";
	key += oa::toString(inDtype);
	
	{
		oa::SharedLock lock(*mutex_);
		
		// Strategy 1: Try with requested DTYPE suffix
		auto it = registry_.find(key);
		if (it != registry_.end()) {
			return it->second;
		}

		// Strategy 2: Preserve legacy FP32 pipelines registered without a DTYPE
		// specialization, but never let one satisfy a different storage ABI.
		// storage.slang uses DTYPE to choose two- versus four-byte addressing, so
		// an inexact variant can read and write outside the tensor layout.
		auto bareIt = registry_.find(oa::String(inName));
		if (bareIt != registry_.end() and bareIt->second.nativeDtype == inDtype) {
			return bareIt->second;
		}
	}
	
	// pipeline not found after all strategies. load the matching embedded SPIR-V
	// lazily when the engine opted out of eager preload (mobile/edge profile).
	if (device_ != nullptr && !inName.empty()) {
		const oa::Status loadStatus = tryLoadOnDemand(*device_, inName, inDtype);
		if (loadStatus.isOk()) {
			oa::SharedLock lock(*mutex_);
			auto it = registry_.find(key);
			if (it != registry_.end()) return it->second;
			auto bareIt = registry_.find(oa::String(inName));
			if (bareIt != registry_.end() and bareIt->second.nativeDtype == inDtype) {
				return bareIt->second;
			}
		}
	}

	// pipeline not found after eager and lazy strategies.
	OaLogWarn(oa::LogComponent::Compute,
		"pipeline not found: '{}' (tried exact DTYPE={} and lazy exact embedded load).",
		oa::String(inName).cStr(), inDtype);
	return sNull;
}
