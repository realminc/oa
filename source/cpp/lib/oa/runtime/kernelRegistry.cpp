#include <oa/runtime/kernelRegistry.h>
#include <oa/core/types.h>
#include <cstring>

// ─── C API implementations ────────────────────────────────────────────────────
// These replace the old inline bodies in KernelRegistry.h.

const oa::ComputeKernel* oa::computeKernelFindByPackedId(oa::U64 inPackedId) {
	for (const auto& k : oa::kernelRegistry::MlKernels) {
		if (k.id == inPackedId) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::VisionKernels) {
		if (k.id == inPackedId) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::UiKernels) {
		if (k.id == inPackedId) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::CryptoKernels) {
		if (k.id == inPackedId) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::AudioKernels) {
		if (k.id == inPackedId) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::RenderKernels) {
		if (k.id == inPackedId) { return &k; }
	}
	return nullptr;
}

const oa::ComputeKernel* oa::computeKernelFindByName(const char* inName) {
	if (!inName) { return nullptr; }
	for (const auto& k : oa::kernelRegistry::MlKernels) {
		if (std::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::VisionKernels) {
		if (std::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::UiKernels) {
		if (std::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::CryptoKernels) {
		if (std::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::AudioKernels) {
		if (std::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::RenderKernels) {
		if (std::strcmp(k.name, inName) == 0) { return &k; }
	}
	return nullptr;
}

bool oa::computeKernelUsesDefaultBindlessPipeline(const char* inName)
{
	// Render shaders (vertex/fragment) use a graphics pipeline layout,
	// not the compute bindless pipeline. Exclude from compute dispatch.
	if (std::strstr(inName, ".vert") || std::strstr(inName, ".frag") ||
	    std::strstr(inName, ".geom") || std::strstr(inName, ".tesc") ||
	    std::strstr(inName, ".tese")) {
		return false;
	}
	// The default bindless pipeline layout now includes storage buffers,
	// storage images, sampled images, and samplers in one update-after-bind set.
	return true;
}
