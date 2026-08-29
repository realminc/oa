#include <oa/runtime/kernelRegistry.h>
#include <oa/core/types.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/stringView.h>

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
		if (oa::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::VisionKernels) {
		if (oa::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::UiKernels) {
		if (oa::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::CryptoKernels) {
		if (oa::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::AudioKernels) {
		if (oa::strcmp(k.name, inName) == 0) { return &k; }
	}
	for (const auto& k : oa::kernelRegistry::RenderKernels) {
		if (oa::strcmp(k.name, inName) == 0) { return &k; }
	}
	return nullptr;
}

bool oa::computeKernelUsesDefaultBindlessPipeline(const char* inName)
{
	if (inName == nullptr) return false;
	// Sampler YCbCr conversion is deliberately hybrid: global bindless set 0
	// plus a decoder-owned immutable combined-image sampler at set 1. Creating
	// either module with the default one-set layout is invalid.
	if (oa::strcmp(inName, "CvtNv12YcbcrToRgba") == 0
		or oa::strcmp(inName, "CvtNv12YcbcrToBf16") == 0) {
		return false;
	}
	// Render shaders (vertex/fragment) use a graphics pipeline layout,
	// not the compute bindless pipeline. Exclude from compute dispatch.
	const oa::StringView name(inName);
	if (name.find(".vert") != oa::StringView::Npos
		|| name.find(".frag") != oa::StringView::Npos
		|| name.find(".geom") != oa::StringView::Npos
		|| name.find(".tesc") != oa::StringView::Npos
		|| name.find(".tese") != oa::StringView::Npos) {
		return false;
	}
	// The default bindless pipeline layout now includes storage buffers,
	// storage images, sampled images, and samplers in one update-after-bind set.
	return true;
}
