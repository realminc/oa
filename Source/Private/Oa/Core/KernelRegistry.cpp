#include <Oa/Core/KernelRegistry.h>
#include <Oa/Core/Extension.h>
#include <Oa/Core/Types.h>
#include <cstring>
#include <cassert>

// ─── Dynamic extension table ─────────────────────────────────────────────────
// Extensions (ml, chain, anim, …) push their kernel metadata here before
// OaEngine::Create loads pipelines. Read-only after engine init.

static OaVec<OaComputeKernel> GDynamicKernels;

namespace OaKernelRegistry {

void RegisterDynamic(OaSpan<const OaComputeKernel> InKernels) {
	for (const auto& k : InKernels) {
		assert(k.Id != 0 && "OaKernelRegistry::RegisterDynamic: kernel ID must be non-zero");
		GDynamicKernels.PushBack(k);
	}
}

} // namespace OaKernelRegistry

// ─── OaExtKernelRegistry ─────────────────────────────────────────────────────

void OaExtKernelRegistry::Add(OaSpan<const OaComputeKernel> InKernels) {
	OaKernelRegistry::RegisterDynamic(InKernels);
}

// ─── C API implementations ────────────────────────────────────────────────────
// These replace the old inline bodies in KernelRegistry.h.
// Search order: built-in Ml → Vision → UI → Crypto → dynamic extension table.

const OaComputeKernel* OaComputeKernelFindByPackedId(OaU64 InPackedId) {
	for (const auto& k : OaKernelRegistry::MlKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::VisionKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::UiKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::CryptoKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::AudioKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::RenderKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	for (const auto& k : GDynamicKernels) {
		if (k.Id == InPackedId) { return &k; }
	}
	return nullptr;
}

const OaComputeKernel* OaComputeKernelFindByName(const char* InName) {
	if (!InName) { return nullptr; }
	for (const auto& k : OaKernelRegistry::MlKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::VisionKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::UiKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::CryptoKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::AudioKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	for (const auto& k : OaKernelRegistry::RenderKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	for (const auto& k : GDynamicKernels) {
		if (std::strcmp(k.Name, InName) == 0) { return &k; }
	}
	return nullptr;
}

bool OaComputeKernelUsesDefaultBindlessPipeline(const char* InName)
{
	// Render shaders (vertex/fragment) use a graphics pipeline layout,
	// not the compute bindless pipeline. Exclude from compute dispatch.
	if (std::strstr(InName, ".vert") || std::strstr(InName, ".frag") ||
	    std::strstr(InName, ".geom") || std::strstr(InName, ".tesc") ||
	    std::strstr(InName, ".tese")) {
		return false;
	}
	// The default bindless pipeline layout now includes storage buffers,
	// storage images, sampled images, and samplers in one update-after-bind set.
	return true;
}
