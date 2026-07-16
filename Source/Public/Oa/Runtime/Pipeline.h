#pragma once

#include <memory>
#include <shared_mutex>

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Spirv.h>

class OaVkDevice;
class OaVkBuffer;

class OaSpecConstant {
public:
	OaU32 Id = 0;
	OaU32 Value = 0;
};

class OaPipelineSpec {
public:
	OaU32 WgSize = 256;
	OaU32 NumBindings = 2;
	OaU32 PushConstantBytes = 128;
	OaVec<OaSpecConstant> SpecConstants;
};

// One self-contained SPIR-V pipeline creation request. Slang source-module
// dependencies are resolved at build time, so runtime preload requests have no
// ordering dependency on one another.
class OaPipelineLoadRequest {
public:
	OaString Name;
	OaSpan<const OaU8> Spirv;
	OaPipelineSpec Spec;
};

class OaComputePipeline {
public:
	void* Pipeline = nullptr;
	void* PipelineLayout = nullptr;
	void* DescriptorSetLayout = nullptr;
	void* DescriptorPool = nullptr;
	void* DescriptorSet = nullptr;
	OaU32 NumBindings = 0;
	OaBool Bindless = false;
	OaU32 NativeDtype = 0;  // 0=FP32, 1=BF16 — the DTYPE this pipeline was compiled with

	[[nodiscard]] static OaResult<OaComputePipeline> Create(
		const OaVkDevice& InDevice,
		OaSpan<const OaU8> InSpirv,
		const OaPipelineSpec& InSpec,
		void* InPipelineCache = nullptr,
		void* InBindlessPipelineLayout = nullptr);
	void Destroy(const OaVkDevice& InDevice);

	[[nodiscard]] OaStatus AllocDescriptorSet(const OaVkDevice& InDevice);
	void WriteStorageBuffer(const OaVkDevice& InDevice, OaU32 InBinding, const OaVkBuffer& InBuffer);
};

class OaPipelineCache {
public:
	void* Cache = nullptr;
	OaU64 InitialDataBytes = 0;

	[[nodiscard]] static OaResult<OaPipelineCache> Create(
		const OaVkDevice& InDevice, const OaString& InCacheDir);
	void Save(const OaVkDevice& InDevice, const OaString& InCacheDir) const;
	void Destroy(const OaVkDevice& InDevice);
};

// Pre-compiled pipeline library — a partial pipeline that can be linked
// into final pipelines at near-zero cost. Requires VK_KHR_pipeline_library.
class OaPipelineLibrary {
public:
	void* Pipeline = nullptr;
	void* PipelineLayout = nullptr;

	[[nodiscard]] static OaResult<OaPipelineLibrary> Create(
		const OaVkDevice& InDevice,
		OaSpan<const OaU8> InSpirv,
		const OaPipelineSpec& InSpec,
		void* InPipelineCache = nullptr,
		void* InBindlessPipelineLayout = nullptr);
	void Destroy(const OaVkDevice& InDevice);

	// Link this library into a final executable pipeline.
	[[nodiscard]] OaResult<OaComputePipeline> Link(
		const OaVkDevice& InDevice,
		void* InPipelineCache = nullptr,
		void* InBindlessPipelineLayout = nullptr) const;
};

class OaPipelineRegistry {
public:
	OaPipelineRegistry() = default;
	~OaPipelineRegistry() = default;
	OaPipelineRegistry(const OaPipelineRegistry&) = delete;
	OaPipelineRegistry& operator=(const OaPipelineRegistry&) = delete;
	// Allow move despite std::mutex member (will use default move which is safe for mutex)
	OaPipelineRegistry(OaPipelineRegistry&&) noexcept = default;
	OaPipelineRegistry& operator=(OaPipelineRegistry&&) noexcept = default;

	[[nodiscard]] OaStatus Init(
		const OaVkDevice& InDevice, const OaString& InCacheDir,
		void* InBindlessPipelineLayout = nullptr);
	void Destroy(const OaVkDevice& InDevice);

	[[nodiscard]] OaStatus EnsurePipeline(
		const OaVkDevice& InDevice,
		OaStringView InName,
		OaSpan<const OaU8> InSpirv,
		const OaPipelineSpec& InSpec);

	// Create independent compute pipelines concurrently. Every worker owns a
	// separate VkPipelineCache because Vulkan requires external synchronization
	// for host access to a pipeline cache. Worker caches are merged into the
	// registry cache after all workers have joined.
	[[nodiscard]] OaStatus EnsurePipelinesParallel(
		const OaVkDevice& InDevice,
		OaSpan<const OaPipelineLoadRequest> InRequests,
		OaU32 InWorkerCount,
		OaVec<OaStatus>* OutStatuses = nullptr);

	[[nodiscard]] bool HasInitialCacheData() const noexcept {
		return Cache_.InitialDataBytes != 0;
	}

	[[nodiscard]] OaComputePipeline& GetPipeline(OaStringView InName, OaU32 InDtype = 0);
	
	// On-demand loading for known BLAS kernels
	[[nodiscard]] OaStatus TryLoadOnDemand(
		const OaVkDevice& InDevice,
		OaStringView InName,
		OaU32 InDtype);

private:
	OaHashMap<OaString, OaComputePipeline> Registry_;
	mutable OaUniquePtr<std::shared_mutex> Mutex_ = OaStdMakeUnique<std::shared_mutex>();
	OaPipelineCache Cache_;
	OaString CacheDir_;
	void* BindlessPipelineLayout_ = nullptr;
	const OaVkDevice* Device_ = nullptr;  // For on-demand loading
};
