#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <thread>
#include <algorithm>
#include <vector>

#include <Oa/Runtime/Pipeline.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Core/FileIo.h>
#include <Oa/Core/Log.h>
#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/EnvFlag.h>

#ifdef __ANDROID__
#include <android/log.h>
#endif


static const char* OA_PIPELINE_CACHE_FILE = "pipeline.vcache";

namespace {
// SIMD-width experiment. OA_GEMM_SUBGROUP_SIZE=8|16|32 pins the compute subgroup
// size on every compute pipeline via VkPipelineShaderStageRequiredSubgroupSize-
// CreateInfo — used to align to the Intel Xe native fp32 vector width (SIMD16),
// which the Intel compiler otherwise often lowers to SIMD8. Holds the struct;
// the caller keeps it alive through vkCreateComputePipelines. Unset → no-op.
struct OaForcedSubgroup {
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo Info{};
	bool Active = false;
};

inline void OaMaybeForceSubgroupSize(VkPipelineShaderStageCreateInfo& InOutStage,
                                     OaForcedSubgroup& OutHolder) {
	const OaI64 sz = OaEnvFlag::GetInt("OA_GEMM_SUBGROUP_SIZE", 0);
	if (sz == 8 || sz == 16 || sz == 32) {
		OutHolder.Info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
		OutHolder.Info.requiredSubgroupSize = static_cast<uint32_t>(sz);
		OutHolder.Info.pNext = nullptr;
		InOutStage.pNext = &OutHolder.Info;   // stage pNext is null on the compute paths here
		OutHolder.Active = true;
	}
}
} // namespace


OaResult<OaPipelineCache> OaPipelineCache::Create(const OaVkDevice& InDevice, const OaString& InCacheDir) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	OaVec<OaU8> cacheData;
	if (!InCacheDir.empty()) {
		OaPath cachePath = OaPath(InCacheDir) / OA_PIPELINE_CACHE_FILE;
		auto loaded = OaFileIo::ReadBinary(cachePath);
		if (loaded.IsOk()) {
			cacheData = std::move(loaded.GetValue());
			OA_LOG_INFO(OaLogComponent::Core, "Pipeline cache: loaded %zu bytes from %s",
				cacheData.Size(), cachePath.string().c_str());
		}
	}

	VkPipelineCacheCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
	ci.initialDataSize = cacheData.Size();
	ci.pInitialData = cacheData.Empty() ? nullptr : cacheData.Data();

	VkPipelineCache cache = VK_NULL_HANDLE;
	VkResult r = vkCreatePipelineCache(dev, &ci, nullptr, &cache);
	if (r != VK_SUCCESS) {
		OA_LOG_WARN(OaLogComponent::Core, "Pipeline cache: creation failed, continuing without cache");
		OaPipelineCache pc;
		return pc;
	}

	OaPipelineCache pc;
	pc.Cache = cache;
	pc.InitialDataBytes = static_cast<OaU64>(cacheData.Size());
	return pc;
}


void OaPipelineCache::Save(const OaVkDevice& InDevice, const OaString& InCacheDir) const {
	if (!Cache || InCacheDir.empty()) {
		return;
	}

	VkDevice dev = static_cast<VkDevice>(InDevice.Device);
	VkPipelineCache cache = static_cast<VkPipelineCache>(Cache);

	size_t dataSize = 0;
	(void)vkGetPipelineCacheData(dev, cache, &dataSize, nullptr);
	if (dataSize == 0) {
		return;
	}

	OaVec<OaU8> data(dataSize);
	(void)vkGetPipelineCacheData(dev, cache, &dataSize, data.Data());

	OaPath cacheDir(InCacheDir);
	(void)OaFileIo::CreateDirectories(cacheDir);
	OaPath cachePath = cacheDir / OA_PIPELINE_CACHE_FILE;
	(void)OaFileIo::WriteBinary(cachePath, OaSpan<const OaU8>(data.Data(), data.Size()));
	OA_LOG_INFO(OaLogComponent::Core, "Pipeline cache: saved %zu bytes to %s",
		dataSize, cachePath.string().c_str());
}

void OaPipelineCache::Destroy(const OaVkDevice& InDevice) {
	if (Cache) {
		VkDevice dev = static_cast<VkDevice>(InDevice.Device);
		vkDestroyPipelineCache(dev, static_cast<VkPipelineCache>(Cache), nullptr);
		Cache = nullptr;
	}
	InitialDataBytes = 0;
}

OaResult<OaComputePipeline> OaComputePipeline::Create(
	const OaVkDevice& InDevice,
	OaSpan<const OaU8> InSpirv,
	const OaPipelineSpec& InSpec,
	void* InPipelineCache,
	void* InBindlessPipelineLayout)
{
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkShaderModuleCreateInfo smCI{};
	smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	smCI.codeSize = InSpirv.size();
	smCI.pCode = reinterpret_cast<const OaU32*>(InSpirv.data());

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	VkResult r = vkCreateShaderModule(dev, &smCI, nullptr, &shaderModule);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::ShaderCompileError, "vkCreateShaderModule failed");
	}

	VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	bool ownsLayout = false;

	if (InBindlessPipelineLayout) {
		// Bindless: shared pipeline layout, no per-pipeline descriptor layout
		pipelineLayout = static_cast<VkPipelineLayout>(InBindlessPipelineLayout);
	} else {
		ownsLayout = true;

		OaVec<VkDescriptorSetLayoutBinding> bindings(InSpec.NumBindings);
		for (OaU32 i = 0; i < InSpec.NumBindings; i++) {
			bindings[i] = {};
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo dslCI{};
		dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslCI.bindingCount = InSpec.NumBindings;
		dslCI.pBindings = bindings.Data();

		r = vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);
		if (r != VK_SUCCESS) {
			vkDestroyShaderModule(dev, shaderModule, nullptr);
			return OaStatus::Error(OaStatusCode::PipelineError, "vkCreateDescriptorSetLayout failed");
		}

		VkPushConstantRange pushRange = {
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = InSpec.PushConstantBytes
		};

		VkPipelineLayoutCreateInfo plCI{};
		plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		plCI.setLayoutCount = 1;
		plCI.pSetLayouts = &dsl;
		plCI.pushConstantRangeCount = (InSpec.PushConstantBytes > 0) ? 1 : 0;
		plCI.pPushConstantRanges = (InSpec.PushConstantBytes > 0) ? &pushRange : nullptr;

		r = vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
		if (r != VK_SUCCESS) {
			vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
			vkDestroyShaderModule(dev, shaderModule, nullptr);
			return OaStatus::Error(OaStatusCode::PipelineError, "vkCreatePipelineLayout failed");
		}
	}

	OaVec<VkSpecializationMapEntry> specEntries;
	OaVec<OaU32> specData;
	VkSpecializationInfo specInfo{};

	if (!InSpec.SpecConstants.Empty()) {
		specEntries.Resize(InSpec.SpecConstants.Size());
		specData.Resize(InSpec.SpecConstants.Size());
		for (OaU32 i = 0; i < InSpec.SpecConstants.Size(); i++) {
			specEntries[i].constantID = InSpec.SpecConstants[i].Id;
			specEntries[i].offset = i * sizeof(OaU32);
			specEntries[i].size = sizeof(OaU32);
			specData[i] = InSpec.SpecConstants[i].Value;
		}
		specInfo.mapEntryCount = static_cast<OaU32>(specEntries.Size());
		specInfo.pMapEntries = specEntries.Data();
		specInfo.dataSize = specData.Size() * sizeof(OaU32);
		specInfo.pData = specData.Data();
	}

	VkComputePipelineCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpCI.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpCI.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	cpCI.stage.module = shaderModule;
	cpCI.stage.pName = "main";
	cpCI.stage.pSpecializationInfo = InSpec.SpecConstants.Empty() ? nullptr : &specInfo;
	cpCI.layout = pipelineLayout;

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(InPipelineCache);
	OaForcedSubgroup forcedSg;
	OaMaybeForceSubgroupSize(cpCI.stage, forcedSg);
	r = vkCreateComputePipelines(dev, vkCache, 1, &cpCI, nullptr, &pipeline);
	if (r != VK_SUCCESS && forcedSg.Active) {
		// Forced subgroup size unsupported for this shader/device — retry
		// unconstrained so the experiment knob can't brick pipeline creation.
		cpCI.stage.pNext = nullptr;
		r = vkCreateComputePipelines(dev, vkCache, 1, &cpCI, nullptr, &pipeline);
	}
	vkDestroyShaderModule(dev, shaderModule, nullptr);
	if (r != VK_SUCCESS) {
		// Bindless path uses engine-owned layout — must not destroy (see Destroy()).
		if (ownsLayout) {
			vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
			vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		}
		return OaStatus::Error(OaStatusCode::PipelineError, "vkCreateComputePipelines failed");
	}

	OaComputePipeline p;
	p.Pipeline = pipeline;
	p.PipelineLayout = pipelineLayout;
	p.DescriptorSetLayout = ownsLayout ? dsl : nullptr;
	p.DescriptorPool = nullptr;
	p.DescriptorSet = nullptr;
	p.NumBindings = InSpec.NumBindings;
	p.Bindless = (InBindlessPipelineLayout != nullptr);
	
	// Extract native DTYPE from spec constants (ID=0 is DTYPE by convention)
	p.NativeDtype = 0;  // Default to FP32
	for (const auto& sc : InSpec.SpecConstants) {
		if (sc.Id == 0) {
			p.NativeDtype = sc.Value;
			break;
		}
	}
	
	return p;
}

void OaComputePipeline::Destroy(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);
	if (DescriptorPool) vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(DescriptorPool), nullptr);
	if (Pipeline) vkDestroyPipeline(dev, static_cast<VkPipeline>(Pipeline), nullptr);
	if (!Bindless) {
		if (PipelineLayout) vkDestroyPipelineLayout(dev, static_cast<VkPipelineLayout>(PipelineLayout), nullptr);
		if (DescriptorSetLayout) vkDestroyDescriptorSetLayout(dev, static_cast<VkDescriptorSetLayout>(DescriptorSetLayout), nullptr);
	}
	Pipeline = PipelineLayout = DescriptorSetLayout = DescriptorPool = DescriptorSet = nullptr;
	Bindless = false;
}

OaStatus OaComputePipeline::AllocDescriptorSet(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	if (DescriptorPool) {
		vkDestroyDescriptorPool(dev, static_cast<VkDescriptorPool>(DescriptorPool), nullptr);
		DescriptorPool = nullptr;
		DescriptorSet = nullptr;
	}

	OaU32 numDesc = NumBindings > 0 ? NumBindings : 16;

	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSize.descriptorCount = numDesc;

	VkDescriptorPoolCreateInfo dpCI{};
	dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	dpCI.maxSets = 1;
	dpCI.poolSizeCount = 1;
	dpCI.pPoolSizes = &poolSize;

	VkDescriptorPool pool = VK_NULL_HANDLE;
	VkResult r = vkCreateDescriptorPool(dev, &dpCI, nullptr, &pool);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::PipelineError, "vkCreateDescriptorPool failed");
	}

	VkDescriptorSetLayout dsl = static_cast<VkDescriptorSetLayout>(DescriptorSetLayout);
	VkDescriptorSetAllocateInfo dsAI{};
	dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	dsAI.descriptorPool = pool;
	dsAI.descriptorSetCount = 1;
	dsAI.pSetLayouts = &dsl;

	VkDescriptorSet ds = VK_NULL_HANDLE;
	r = vkAllocateDescriptorSets(dev, &dsAI, &ds);
	if (r != VK_SUCCESS) {
		vkDestroyDescriptorPool(dev, pool, nullptr);
		return OaStatus::Error(OaStatusCode::PipelineError, "vkAllocateDescriptorSets failed");
	}

	DescriptorPool = pool;
	DescriptorSet = ds;
	return OaStatus::Ok();
}

void OaComputePipeline::WriteStorageBuffer(
	const OaVkDevice& InDevice,
	OaU32 InBinding,
	const OaVkBuffer& InBuffer)
{
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkDescriptorBufferInfo bufInfo{};
	bufInfo.buffer = static_cast<VkBuffer>(InBuffer.Buffer);
	bufInfo.offset = 0;
	bufInfo.range = InBuffer.Size;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = static_cast<VkDescriptorSet>(DescriptorSet);
	write.dstBinding = InBinding;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufInfo;

	vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
}

// ─── Pipeline Library ──────────────────────────────────────────────────────

OaResult<OaPipelineLibrary> OaPipelineLibrary::Create(
	const OaVkDevice& InDevice,
	OaSpan<const OaU8> InSpirv,
	const OaPipelineSpec& InSpec,
	void* InPipelineCache,
	void* InBindlessPipelineLayout)
{
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkShaderModuleCreateInfo smCI{};
	smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	smCI.codeSize = InSpirv.size();
	smCI.pCode = reinterpret_cast<const OaU32*>(InSpirv.data());

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	VkResult r = vkCreateShaderModule(dev, &smCI, nullptr, &shaderModule);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::ShaderCompileError,
			"pipeline library: vkCreateShaderModule failed");
	}

	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	if (InBindlessPipelineLayout) {
		pipelineLayout = static_cast<VkPipelineLayout>(InBindlessPipelineLayout);
	} else {
		OaVec<VkDescriptorSetLayoutBinding> bindings(InSpec.NumBindings);
		for (OaU32 i = 0; i < InSpec.NumBindings; i++) {
			bindings[i] = {};
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		}

		VkDescriptorSetLayoutCreateInfo dslCI{};
		dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslCI.bindingCount = InSpec.NumBindings;
		dslCI.pBindings = bindings.Data();

		VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
		r = vkCreateDescriptorSetLayout(dev, &dslCI, nullptr, &dsl);
		if (r != VK_SUCCESS) {
			vkDestroyShaderModule(dev, shaderModule, nullptr);
			return OaStatus::Error(OaStatusCode::PipelineError,
				"pipeline library: vkCreateDescriptorSetLayout failed");
		}

		VkPushConstantRange pushRange = {
			.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
			.offset = 0,
			.size = InSpec.PushConstantBytes
		};

		VkPipelineLayoutCreateInfo plCI{};
		plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		plCI.setLayoutCount = 1;
		plCI.pSetLayouts = &dsl;
		plCI.pushConstantRangeCount = (InSpec.PushConstantBytes > 0) ? 1 : 0;
		plCI.pPushConstantRanges = (InSpec.PushConstantBytes > 0) ? &pushRange : nullptr;

		r = vkCreatePipelineLayout(dev, &plCI, nullptr, &pipelineLayout);
		vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
		if (r != VK_SUCCESS) {
			vkDestroyShaderModule(dev, shaderModule, nullptr);
			return OaStatus::Error(OaStatusCode::PipelineError,
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

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(InPipelineCache);
	OaForcedSubgroup forcedSg;
	OaMaybeForceSubgroupSize(cpCI.stage, forcedSg);
	r = vkCreateComputePipelines(dev, vkCache, 1, &cpCI, nullptr, &pipeline);
	if (r != VK_SUCCESS && forcedSg.Active) {
		cpCI.stage.pNext = nullptr;   // retry unconstrained (see OaComputePipeline::Create)
		r = vkCreateComputePipelines(dev, vkCache, 1, &cpCI, nullptr, &pipeline);
	}
	vkDestroyShaderModule(dev, shaderModule, nullptr);
	if (r != VK_SUCCESS) {
		if (!InBindlessPipelineLayout) {
			vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
		}
		return OaStatus::Error(OaStatusCode::PipelineError,
			"pipeline library: vkCreateComputePipelines (LIBRARY_BIT) failed");
	}

	OaPipelineLibrary lib;
	lib.Pipeline = pipeline;
	lib.PipelineLayout = pipelineLayout;
	return lib;
}

void OaPipelineLibrary::Destroy(const OaVkDevice& InDevice) {
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);
	if (Pipeline) {
		vkDestroyPipeline(dev, static_cast<VkPipeline>(Pipeline), nullptr);
		Pipeline = nullptr;
	}
}

OaResult<OaComputePipeline> OaPipelineLibrary::Link(
	const OaVkDevice& InDevice,
	void* InPipelineCache,
	void* InBindlessPipelineLayout) const
{
	VkDevice dev = static_cast<VkDevice>(InDevice.Device);

	VkPipeline lib = static_cast<VkPipeline>(Pipeline);
	VkPipelineLibraryCreateInfoKHR libInfo{};
	libInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
	libInfo.libraryCount = 1;
	libInfo.pLibraries = &lib;

	VkComputePipelineCreateInfo cpCI{};
	cpCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	cpCI.pNext = &libInfo;
	cpCI.layout = static_cast<VkPipelineLayout>(PipelineLayout);

	VkPipeline pipeline = VK_NULL_HANDLE;
	VkPipelineCache vkCache = static_cast<VkPipelineCache>(InPipelineCache);
	VkResult r = vkCreateComputePipelines(dev, vkCache, 1, &cpCI, nullptr, &pipeline);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::PipelineError,
			"pipeline library: link failed");
	}

	OaComputePipeline p;
	p.Pipeline = pipeline;
	p.PipelineLayout = static_cast<VkPipelineLayout>(PipelineLayout);
	p.Bindless = (InBindlessPipelineLayout != nullptr);
	return p;
}

// ─── Pipeline Registry ────────────────────────────────────────────────────

namespace {
OaString OaMakePipelineKey(OaStringView InName, const OaPipelineSpec& InSpec) {
	OaString key(InName);
	for (const auto& sc : InSpec.SpecConstants) {
		key += "|";
		key += OaToString(sc.Id);
		key += "=";
		key += OaToString(sc.Value);
	}
	return key;
}
} // namespace

OaStatus OaPipelineRegistry::Init(
	const OaVkDevice& InDevice, const OaString& InCacheDir,
	void* InBindlessPipelineLayout)
{
	Device_ = &InDevice;
	CacheDir_ = InCacheDir;
	BindlessPipelineLayout_ = InBindlessPipelineLayout;
	auto cacheResult = OaPipelineCache::Create(InDevice, InCacheDir);
	if (cacheResult.IsOk()) Cache_ = cacheResult.GetValue();
	return OaStatus::Ok();
}

void OaPipelineRegistry::Destroy(const OaVkDevice& InDevice) {
	Cache_.Save(InDevice, CacheDir_);
	Cache_.Destroy(InDevice);
	std::unique_lock lock(*Mutex_);
	for (auto& [name, pipe] : Registry_) {
		pipe.Destroy(InDevice);
	}
	Registry_.Clear();
	Device_ = nullptr;  // Clear dangling pointer to prevent use-after-free
}

OaStatus OaPipelineRegistry::EnsurePipeline(
	const OaVkDevice& InDevice,
	OaStringView InName,
	OaSpan<const OaU8> InSpirv,
	const OaPipelineSpec& InSpec)
{
	OaString key = OaMakePipelineKey(InName, InSpec);

	{
		std::shared_lock lock(*Mutex_);
		if (Registry_.Contains(key)) return OaStatus::Ok();
	}

	// Direct creation + pipeline cache. OaPipelineLibrary::Create/Link is
	// available for manual use but not auto-integrated here -- pipeline
	// libraries target multi-stage (ray tracing / graphics) pipelines;
	// for single-stage compute the VkPipelineCache already provides
	// equivalent cold-start reduction.
	auto result = OaComputePipeline::Create(
		InDevice, InSpirv, InSpec, Cache_.Cache, BindlessPipelineLayout_);
	if (!result.IsOk() && Cache_.Cache) {
		OA_LOG_WARN(OaLogComponent::Core,
			"Pipeline creation failed with cache for %s, dropping %s and retrying uncached",
			OaString(InName).c_str(), OA_PIPELINE_CACHE_FILE);
		Cache_.Destroy(InDevice);
		if (!CacheDir_.empty()) {
			(void)OaFileIo::RemoveFile(OaPath(CacheDir_) / OA_PIPELINE_CACHE_FILE);
		}
		result = OaComputePipeline::Create(
			InDevice, InSpirv, InSpec, nullptr, BindlessPipelineLayout_);
	}
	if (!result.IsOk()) return result.GetStatus();

	std::unique_lock lock(*Mutex_);
	if (Registry_.Contains(key)) {
		result.GetValue().Destroy(InDevice);
		return OaStatus::Ok();
	}
	Registry_.Emplace(std::move(key), std::move(result.GetValue()));
	return OaStatus::Ok();
}

OaStatus OaPipelineRegistry::EnsurePipelinesParallel(
	const OaVkDevice& InDevice,
	OaSpan<const OaPipelineLoadRequest> InRequests,
	OaU32 InWorkerCount,
	OaVec<OaStatus>* OutStatuses)
{
	const OaU32 requestCount = static_cast<OaU32>(InRequests.size());
	if (OutStatuses) {
		OutStatuses->Resize(requestCount);
	}
	if (requestCount == 0) return OaStatus::Ok();

	const OaU32 workerCount = std::max<OaU32>(1u,
		std::min<OaU32>(InWorkerCount, requestCount));
	if (workerCount == 1) {
		OaStatus firstError = OaStatus::Ok();
		for (OaU32 i = 0; i < requestCount; ++i) {
			const auto& request = InRequests[i];
			OaStatus status = EnsurePipeline(
				InDevice, request.Name, request.Spirv, request.Spec);
			if (OutStatuses) (*OutStatuses)[i] = status;
			if (firstError.IsOk() && status.IsError()) firstError = status;
		}
		return firstError;
	}

	class OaParallelBuildResult {
	public:
		OaStatus Status;
		OaComputePipeline Pipeline;
		bool NeedsBuild = true;
		bool HasPipeline = false;
	};

	OaVec<OaParallelBuildResult> results(requestCount);
	{
		std::shared_lock lock(*Mutex_);
		for (OaU32 i = 0; i < requestCount; ++i) {
			if (Registry_.Contains(OaMakePipelineKey(InRequests[i].Name, InRequests[i].Spec))) {
				results[i].NeedsBuild = false;
			}
		}
	}

	VkDevice device = static_cast<VkDevice>(InDevice.Device);
	VkPipelineCache primaryCache = static_cast<VkPipelineCache>(Cache_.Cache);

	// Snapshot the primary cache once, before workers start. Feeding this same
	// immutable snapshot to each worker preserves warm-cache startup while keeping
	// every vkCreateComputePipelines call on a separately synchronized cache.
	OaVec<OaU8> initialData;
	if (primaryCache != VK_NULL_HANDLE) {
		size_t size = 0;
		if (vkGetPipelineCacheData(device, primaryCache, &size, nullptr) == VK_SUCCESS && size != 0) {
			initialData.Resize(size);
			if (vkGetPipelineCacheData(device, primaryCache, &size, initialData.Data()) != VK_SUCCESS) {
				initialData.Clear();
			} else if (size < initialData.Size()) {
				initialData.Resize(size);
			}
		}
	}

	OaVec<VkPipelineCache> workerCaches(workerCount);
	for (OaU32 i = 0; i < workerCount; ++i) {
		VkPipelineCacheCreateInfo ci{};
		ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		ci.initialDataSize = initialData.Size();
		ci.pInitialData = initialData.Empty() ? nullptr : initialData.Data();
		if (vkCreatePipelineCache(device, &ci, nullptr, &workerCaches[i]) != VK_SUCCESS) {
			workerCaches[i] = VK_NULL_HANDLE;
			OA_LOG_WARN(OaLogComponent::Core,
				"Shader preload worker %u: pipeline-cache creation failed; compiling uncached", i);
		}
	}

	std::atomic<OaU32> nextIndex{0};
	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (OaU32 worker = 0; worker < workerCount; ++worker) {
		workers.emplace_back([&, worker] {
			for (;;) {
				const OaU32 index = nextIndex.fetch_add(1, std::memory_order_relaxed);
				if (index >= requestCount) break;
				auto& build = results[index];
				if (!build.NeedsBuild) continue;

				const auto& request = InRequests[index];
				auto result = OaComputePipeline::Create(
					InDevice, request.Spirv, request.Spec,
					workerCaches[worker], BindlessPipelineLayout_);
				if (!result.IsOk() && workerCaches[worker] != VK_NULL_HANDLE) {
					// A corrupt/incompatible cache entry must not make the shader unloadable.
					// Retry this pipeline without mutating the shared primary cache.
					result = OaComputePipeline::Create(
						InDevice, request.Spirv, request.Spec,
						nullptr, BindlessPipelineLayout_);
				}
				if (!result.IsOk()) {
					build.Status = result.GetStatus();
					continue;
				}
				build.Pipeline = std::move(result.GetValue());
				build.HasPipeline = true;
			}
		});
	}
	for (auto& worker : workers) worker.join();

	// Merge only after all workers stop touching their caches. This satisfies the
	// Vulkan external-synchronization requirements for both source and destination.
	if (primaryCache != VK_NULL_HANDLE) {
		OaVec<VkPipelineCache> mergeCaches;
		mergeCaches.Reserve(workerCount);
		for (VkPipelineCache cache : workerCaches) {
			if (cache != VK_NULL_HANDLE) mergeCaches.PushBack(cache);
		}
		if (!mergeCaches.Empty()) {
			const VkResult merged = vkMergePipelineCaches(
				device, primaryCache, static_cast<OaU32>(mergeCaches.Size()), mergeCaches.Data());
			if (merged != VK_SUCCESS) {
				OA_LOG_WARN(OaLogComponent::Core,
					"Shader preload: vkMergePipelineCaches failed (VkResult=%d)",
					static_cast<int>(merged));
			}
		}
	}
	for (VkPipelineCache cache : workerCaches) {
		if (cache != VK_NULL_HANDLE) vkDestroyPipelineCache(device, cache, nullptr);
	}

	OaStatus firstError = OaStatus::Ok();
	{
		std::unique_lock lock(*Mutex_);
		for (OaU32 i = 0; i < requestCount; ++i) {
			auto& build = results[i];
			if (build.HasPipeline) {
				OaString key = OaMakePipelineKey(InRequests[i].Name, InRequests[i].Spec);
				if (Registry_.Contains(key)) {
					build.Pipeline.Destroy(InDevice);
				} else {
					Registry_.Emplace(std::move(key), std::move(build.Pipeline));
				}
			}
			if (OutStatuses) (*OutStatuses)[i] = build.Status;
			if (firstError.IsOk() && build.Status.IsError()) firstError = build.Status;
		}
	}
	return firstError;
}

OaStatus OaPipelineRegistry::TryLoadOnDemand(
	const OaVkDevice& InDevice,
	OaStringView InName,
	OaU32 InDtype)
{
	OaString kernelName(InName);
	if (!OaComputeKernelUsesDefaultBindlessPipeline(kernelName.c_str())) {
		return OaStatus::NotFound("kernel uses a non-default image/presentation pipeline layout");
	}

	// Look up SPIR-V
	auto* spirv = OaSpvFindAny(kernelName.c_str());
	if (!spirv) {
		OA_LOG_WARN(OaLogComponent::Core, "TryLoadOnDemand: SPIR-V not found for '%s'", kernelName.c_str());
		return OaStatus::NotFound("SPIR-V not found in registry");
	}
	
	OA_LOG_INFO(OaLogComponent::Core, "TryLoadOnDemand: Loading '%s' with DTYPE=%u (spirv size=%u)",
		kernelName.c_str(), InDtype, spirv->Size);
	#ifdef __ANDROID__
	__android_log_print(ANDROID_LOG_INFO, "OA", "Loading pipeline %s dtype=%u (%u bytes)",
		kernelName.c_str(), InDtype, spirv->Size);
	#endif
	
	// Match the embedded preload ABI exactly. All default bindless shaders share
	// the 16-slot heap layout and 128-byte push range; DTYPE is specialization
	// constant 0. Keeping lazy and eager creation identical is essential because
	// graph nodes resolve the variant by their operand dtype.
	OaPipelineSpec spec;
	spec.NumBindings = 16;
	spec.PushConstantBytes = 128;
	spec.WgSize = kernelName.find("CoopMat") != OaString::npos ? 128 : 256;
	spec.SpecConstants = {{.Id = 0, .Value = InDtype}};

	// Load it
	OaStatus status = EnsurePipeline(InDevice, kernelName,
		OaSpan<const OaU8>(spirv->Data, spirv->Size), spec);
	
	if (status.IsOk()) {
		#ifdef __ANDROID__
		__android_log_print(ANDROID_LOG_INFO, "OA", "Loaded pipeline %s dtype=%u",
			kernelName.c_str(), InDtype);
		#endif
		OA_LOG_INFO(OaLogComponent::Core, "TryLoadOnDemand: Successfully loaded '%s' with DTYPE=%u",
			kernelName.c_str(), InDtype);
	} else {
		OA_LOG_WARN(OaLogComponent::Core, "TryLoadOnDemand: Failed to load '%s'",
			kernelName.c_str());
	}
	
	return status;
}

OaComputePipeline& OaPipelineRegistry::GetPipeline(OaStringView InName, OaU32 InDtype) {
	// Single static null pipeline - must be outside all scopes
	static OaComputePipeline sNull;
	
	OaString key(InName);
	key += "|0=";
	key += OaToString(InDtype);
	
	{
		std::shared_lock lock(*Mutex_);
		
		// Strategy 1: Try with requested DTYPE suffix
		auto it = Registry_.Find(key);
		if (it != Registry_.End()) {
			return it->second;
		}
		
		// Strategy 2: Try opposite DTYPE (BF16 kernels called from FP32 engine, or vice versa)
		OaU32 oppositeDtype = (InDtype == 0) ? 1 : 0;
		OaString oppositeKey(InName);
		oppositeKey += "|0=";
		oppositeKey += OaToString(oppositeDtype);
		auto oppositeIt = Registry_.Find(oppositeKey);
		if (oppositeIt != Registry_.End()) {
			return oppositeIt->second;
		}
		
		// Strategy 3: Try bare name without DTYPE suffix (legacy/precision-agnostic kernels)
		auto bareIt = Registry_.Find(OaString(InName));
		if (bareIt != Registry_.End()) {
			return bareIt->second;
		}
	}
	
	// Pipeline not found after all strategies. Load the matching embedded SPIR-V
	// lazily when the engine opted out of eager preload (mobile/edge profile).
	if (Device_ != nullptr && !InName.Empty()) {
		const OaStatus loadStatus = TryLoadOnDemand(*Device_, InName, InDtype);
		if (loadStatus.IsOk()) {
			std::shared_lock lock(*Mutex_);
			auto it = Registry_.Find(key);
			if (it != Registry_.End()) return it->second;
			auto bareIt = Registry_.Find(OaString(InName));
			if (bareIt != Registry_.End()) return bareIt->second;
		}
	}

	// Pipeline not found after eager and lazy strategies.
	OA_LOG_WARN(OaLogComponent::Core,
		"Pipeline not found: '%s' (tried DTYPE=%u, opposite DTYPE, bare name, lazy embedded load).",
		OaString(InName).c_str(), InDtype);
	return sNull;
}
