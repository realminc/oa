#pragma once

#include <memory>
#include <shared_mutex>

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/spirv.h>

namespace oavk { class Device; }
namespace oavk { class Buffer; }

namespace oa {

class SpecConstant {
public:
	oa::U32 id = 0;
	oa::U32 value = 0;
};

class PipelineSpec {
public:
	oa::U32 numBindings = 2;
	oa::U32 pushConstantBytes = 128;
	oa::Vec<oa::SpecConstant> specConstants;
};

// One self-contained SPIR-V pipeline creation request. slang source-module
// dependencies are resolved at build time, so runtime preload requests have no
// ordering dependency on one another.
class PipelineLoadRequest {
public:
	oa::String name;
	oa::Span<const oa::U8> spirv;
	oa::PipelineSpec spec;
};

class PipelineVariantRequest {
public:
	oa::String name;
	oa::U32 dtype = 0;
};

class ComputePipeline {
public:
	void* pipeline = nullptr;
	void* pipelineLayout = nullptr;
	void* descriptorSetLayout = nullptr;
	void* descriptorPool = nullptr;
	void* descriptorSet = nullptr;
	oa::U32 numBindings = 0;
	oa::Bool bindless = false;
	oa::U32 nativeDtype = 0;  // 0=FP32, 1=BF16 — the DTYPE this pipeline was compiled with

	[[nodiscard]] static oa::Result<oa::ComputePipeline> create(
		const oavk::Device& inDevice,
		oa::Span<const oa::U8> inSpirv,
		const oa::PipelineSpec& inSpec,
		void* inPipelineCache = nullptr,
		void* inBindlessPipelineLayout = nullptr);
	void destroy(const oavk::Device& inDevice);

	[[nodiscard]] oa::Status allocDescriptorSet(const oavk::Device& inDevice);
	[[nodiscard]] oa::Status writeStorageBuffer(
		const oavk::Device& inDevice,
		oa::U32 inBinding,
		const oavk::Buffer& inBuffer);
};

class PipelineCache {
public:
	void* cache = nullptr;
	oa::U64 initialDataBytes = 0;

	[[nodiscard]] static oa::Result<oa::PipelineCache> create(
		const oavk::Device& inDevice, const oa::String& inCacheDir);
	void save(const oavk::Device& inDevice, const oa::String& inCacheDir) const;
	void destroy(const oavk::Device& inDevice);
};

// Pre-compiled pipeline library — a partial pipeline that can be linked
// into final pipelines at near-zero cost. Requires VK_KHR_pipeline_library.
class PipelineLibrary {
public:
	void* pipeline = nullptr;
	void* pipelineLayout = nullptr;

	[[nodiscard]] static oa::Result<oa::PipelineLibrary> create(
		const oavk::Device& inDevice,
		oa::Span<const oa::U8> inSpirv,
		const oa::PipelineSpec& inSpec,
		void* inPipelineCache = nullptr,
		void* inBindlessPipelineLayout = nullptr);
	void destroy(const oavk::Device& inDevice);

	// Link this library into a final executable pipeline.
	[[nodiscard]] oa::Result<oa::ComputePipeline> link(
		const oavk::Device& inDevice,
		void* inPipelineCache = nullptr,
		void* inBindlessPipelineLayout = nullptr) const;
};

class PipelineRegistry {
public:
	PipelineRegistry() = default;
	~PipelineRegistry() = default;
	PipelineRegistry(const PipelineRegistry&) = delete;
	PipelineRegistry& operator=(const PipelineRegistry&) = delete;
	// Allow move despite std::mutex member (will use default move which is safe for mutex)
	PipelineRegistry(PipelineRegistry&&) noexcept = default;
	PipelineRegistry& operator=(PipelineRegistry&&) noexcept = default;

	[[nodiscard]] oa::Status init(
		const oavk::Device& inDevice, const oa::String& inCacheDir,
		void* inBindlessPipelineLayout = nullptr);
	void destroy(const oavk::Device& inDevice);

	[[nodiscard]] oa::Status ensurePipeline(
		const oavk::Device& inDevice,
		oa::StringView inName,
		oa::Span<const oa::U8> inSpirv,
		const oa::PipelineSpec& inSpec);

	// Create independent compute pipelines concurrently. Every worker owns a
	// separate VkPipelineCache because vulkan requires external synchronization
	// for host access to a pipeline cache. Worker caches are merged into the
	// registry cache after all workers have joined.
	[[nodiscard]] oa::Status ensurePipelinesParallel(
		const oavk::Device& inDevice,
		oa::Span<const oa::PipelineLoadRequest> inRequests,
		oa::U32 inWorkerCount,
		oa::Vec<oa::Status>* outStatuses = nullptr);
	[[nodiscard]] oa::Status ensurePipelinesOnDemand(
		oa::Span<const oa::PipelineVariantRequest> inRequests);

	[[nodiscard]] bool hasInitialCacheData() const noexcept {
		return cache_.initialDataBytes != 0;
	}

	[[nodiscard]] oa::ComputePipeline& getPipeline(oa::StringView inName, oa::U32 inDtype);
	
	// Fallback single-variant loading for dispatch paths that do not compile an
	// executable graph. Graph compilation uses ensurePipelinesOnDemand instead.
	[[nodiscard]] oa::Status tryLoadOnDemand(
		const oavk::Device& inDevice,
		oa::StringView inName,
		oa::U32 inDtype);

private:
	oa::HashMap<oa::String, oa::ComputePipeline> registry_;
	mutable oa::UniquePtr<std::shared_mutex> mutex_ = oa::makeUnique<std::shared_mutex>();
	oa::PipelineCache cache_;
	oa::String cacheDir_;
	void* bindlessPipelineLayout_ = nullptr;
	const oavk::Device* device_ = nullptr;  // For on-demand loading
};

} // namespace oa
