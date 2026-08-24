#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <algorithm>
#include <vector>

#include <oa/runtime/pipeline.h>
#include "descriptorValidation.h"
#include <oa/runtime/device.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/oaVk.h>
#include <oa/core/filesystem.h>
#include <oa/core/log.h>
#include "../core/logAccess.h"
#include <oa/runtime/kernelRegistry.h>
#include <oa/core/envFlag.h>

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

	oa::Vec<oa::U8> cacheData;
	if (!inCacheDir.empty()) {
		oa::Path cachePath = oa::Path(inCacheDir) / pipelineCacheFile;
		auto loaded = oa::Filesystem::readBinary(cachePath);
		if (loaded.isOk()) {
			cacheData = std::move(loaded.getValue());
			OaLogInfo(oa::LogComponent::Compute, "pipeline cache: loaded %zu bytes from %s",	cacheData.size(), cachePath.string().cStr());
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

	oa::Vec<oa::U8> data(dataSize);
	(void)inDevice.deviceDispatch.vkGetPipelineCacheData(dev, vkCache, &dataSize, data.data());

	oa::Path cacheDir(inCacheDir);
	(void)oa::Filesystem::createDirectories(cacheDir);
	oa::Path cachePath = cacheDir / pipelineCacheFile;
	(void)oa::Filesystem::writeBinary(cachePath, oa::Span<const oa::U8>(data.data(), data.size()));
	OaLogInfo(oa::LogComponent::Compute, "pipeline cache: saved %zu bytes to %s",
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

		oa::Vec<VkDescriptorSetLayoutBinding> bindings(inSpec.numBindings);
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

	oa::Vec<VkSpecializationMapEntry> specEntries;
	oa::Vec<oa::U32> specData;
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
		oa::Vec<VkDescriptorSetLayoutBinding> bindings(inSpec.numBindings);
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
	std::unique_lock lock(*mutex_);
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
		std::shared_lock lock(*mutex_);
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
			"pipeline creation failed with cache for %s, dropping %s and retrying uncached",
			oa::String(inName).cStr(), pipelineCacheFile);
		cache_.destroy(inDevice);
		if (!cacheDir_.empty()) {
			(void)oa::Filesystem::removeFile(oa::Path(cacheDir_) / pipelineCacheFile);
		}
		result = oa::ComputePipeline::create(
			inDevice, inSpirv, inSpec, nullptr, bindlessPipelineLayout_);
	}
	if (!result.isOk()) return result.getStatus();

	std::unique_lock lock(*mutex_);
	if (registry_.contains(key)) {
		result.getValue().destroy(inDevice);
		return oa::Status::ok();
	}
	registry_.emplace(std::move(key), std::move(result.getValue()));
	return oa::Status::ok();
}

oa::Status oa::PipelineRegistry::ensurePipelinesParallel(
	const oavk::Device& inDevice,
	oa::Span<const oa::PipelineLoadRequest> inRequests,
	oa::U32 inWorkerCount,
	oa::Vec<oa::Status>* outStatuses)
{
	const oa::U32 requestCount = static_cast<oa::U32>(inRequests.size());
	if (outStatuses) {
		outStatuses->resize(requestCount);
	}
	if (requestCount == 0) return oa::Status::ok();

	const oa::U32 workerCount = std::max<oa::U32>(1u,
		std::min<oa::U32>(inWorkerCount, requestCount));
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

	oa::Vec<ParallelBuildResult> results(requestCount);
	{
		std::shared_lock lock(*mutex_);
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
	oa::Vec<oa::U8> initialData;
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

	oa::Vec<VkPipelineCache> workerCaches(workerCount);
	for (oa::U32 i = 0; i < workerCount; ++i) {
		VkPipelineCacheCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		ci.initialDataSize = initialData.size();
		ci.pInitialData = initialData.empty() ? nullptr : initialData.data();
		if (inDevice.deviceDispatch.vkCreatePipelineCache(device, &ci, nullptr, &workerCaches[i]) != VK_SUCCESS) {
			workerCaches[i] = VK_NULL_HANDLE;
			OaLogWarn(oa::LogComponent::Compute,
				"shader preload worker %u: pipeline-cache creation failed; compiling uncached", i);
		}
	}

	std::atomic<oa::U32> nextIndex{0};
	const oa::LogSelection logSelection = oa::LogAccess::currentSelection();
	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (oa::U32 worker = 0; worker < workerCount; ++worker) {
		workers.emplace_back([&, worker] {
			oa::LogAccess::Scope logScope(logSelection);
			for (;;) {
				const oa::U32 index = nextIndex.fetch_add(1, std::memory_order_relaxed);
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
				build.pipeline = std::move(result.getValue());
				build.hasPipeline = true;
			}
		});
	}
	for (auto& worker : workers) worker.join();

	// Merge only after all workers stop touching their caches. This satisfies the
	// vulkan external-synchronization requirements for both source and destination.
	if (primaryCache != VK_NULL_HANDLE) {
		oa::Vec<VkPipelineCache> mergeCaches;
		mergeCaches.reserve(workerCount);
		for (VkPipelineCache cache : workerCaches) {
			if (cache != VK_NULL_HANDLE) mergeCaches.pushBack(cache);
		}
		if (!mergeCaches.empty()) {
			const VkResult merged = inDevice.deviceDispatch.vkMergePipelineCaches(
				device, primaryCache, static_cast<oa::U32>(mergeCaches.size()), mergeCaches.data());
			if (merged != VK_SUCCESS) {
				OaLogWarn(oa::LogComponent::Compute,
					"shader preload: vkMergePipelineCaches failed (VkResult=%d)",
					static_cast<int>(merged));
			}
		}
	}
	for (VkPipelineCache cache : workerCaches) {
		if (cache != VK_NULL_HANDLE) inDevice.deviceDispatch.vkDestroyPipelineCache(device, cache, nullptr);
	}

	oa::Status firstError = oa::Status::ok();
	{
		std::unique_lock lock(*mutex_);
		for (oa::U32 i = 0; i < requestCount; ++i) {
			auto& build = results[i];
			if (build.hasPipeline) {
				oa::String key = makePipelineKey(inRequests[i].name, inRequests[i].spec);
				if (registry_.contains(key)) {
					build.pipeline.destroy(inDevice);
				} else {
					registry_.emplace(std::move(key), std::move(build.pipeline));
				}
			}
			if (outStatuses) (*outStatuses)[i] = build.status;
			if (firstError.isOk() && build.status.isError()) firstError = build.status;
		}
	}
	return firstError;
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
		OaLogWarn(oa::LogComponent::Compute, "TryLoadOnDemand: SPIR-V not found for '%s'", kernelName.cStr());
		return oa::Status::notFound("SPIR-V not found in registry");
	}
	
	OaLogInfo(oa::LogComponent::Compute, "TryLoadOnDemand: Loading '%s' with DTYPE=%u (spirv size=%u)",
		kernelName.cStr(), inDtype, spirv->size);
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
	spec.specConstants = {{.id = 0, .value = inDtype}};

	// load it
	oa::Status status = ensurePipeline(inDevice, kernelName,
		oa::Span<const oa::U8>(spirv->data, spirv->size), spec);
	
	if (status.isOk()) {
		#ifdef __ANDROID__
		__android_log_print(ANDROID_LOG_INFO, "OA", "Loaded pipeline %s dtype=%u",
			kernelName.cStr(), inDtype);
		#endif
		OaLogInfo(oa::LogComponent::Compute, "TryLoadOnDemand: Successfully loaded '%s' with DTYPE=%u",
			kernelName.cStr(), inDtype);
	} else {
		OaLogWarn(oa::LogComponent::Compute, "TryLoadOnDemand: Failed to load '%s'",
			kernelName.cStr());
	}
	
	return status;
}

oa::ComputePipeline& oa::PipelineRegistry::getPipeline(oa::StringView inName, oa::U32 inDtype) {
	// Single static null pipeline - must be outside all scopes
	static oa::ComputePipeline sNull;
	if (inDtype > 1U) {
		OaLogError(oa::LogComponent::Compute,
			"pipeline lookup rejected invalid storage DTYPE=%u for '%s'",
			inDtype, oa::String(inName).cStr());
		return sNull;
	}
	
	oa::String key(inName);
	key += "|0=";
	key += oa::toString(inDtype);
	
	{
		std::shared_lock lock(*mutex_);
		
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
			std::shared_lock lock(*mutex_);
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
		"pipeline not found: '%s' (tried exact DTYPE=%u and lazy exact embedded load).",
		oa::String(inName).cStr(), inDtype);
	return sNull;
}
