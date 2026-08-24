#pragma once

#include <oa/runtime/computeKernel.h>

// OA's fixed registry is a composition of three non-overlapping authorities:
// standalone kernels, operation-owned kernels, and Tile GEMM variants. The
// generated configured SPIR-V output list remains the exact embedding/runtime
// manifest; Tools/audit_kernels.py proves the composition is bijective.
namespace oa::kernelRegistry {

struct KernelIdReservation {
	oa::U32 prefix;
	oa::U32 firstLocal;
	oa::U32 lastLocal;
};

static constexpr KernelIdReservation ReservedKernelIdRanges[] = {
#include <oa/runtime/gen/kernelReservations.inl>
};

static constexpr oa::ComputeKernel MlKernels[] = {
#include <oa/runtime/gen/kernelRegistryStandaloneMl.inl>
#include <oa/runtime/gen/tileKernelRegistry.inc>
#include <oa/runtime/gen/kernelRegistryMl.inl>
};

static constexpr oa::ComputeKernel VisionKernels[] = {
#include <oa/runtime/gen/kernelRegistryStandaloneVision.inl>
#include <oa/runtime/gen/kernelRegistryVision.inl>
};

static constexpr oa::ComputeKernel UiKernels[] = {
#include <oa/runtime/gen/kernelRegistryStandaloneUi.inl>
#include <oa/runtime/gen/kernelRegistryUi.inl>
};

static constexpr oa::ComputeKernel AudioKernels[] = {
#include <oa/runtime/gen/kernelRegistryStandaloneAudio.inl>
};

static constexpr oa::ComputeKernel RenderKernels[] = {
#include <oa/runtime/gen/kernelRegistryStandaloneRender.inl>
};

static constexpr oa::ComputeKernel CryptoKernels[] = {
#include <oa/runtime/gen/kernelRegistryStandaloneCrypto.inl>
};

template <typename T, oa::Usize N>
constexpr oa::Span<const T> spanOf(const T (&inValues)[N]) {
	return oa::Span<const T>(inValues, N);
}

inline oa::Span<const oa::ComputeKernel> getMlKernels() { return spanOf(MlKernels); }
inline oa::Span<const oa::ComputeKernel> getVisionKernels() { return spanOf(VisionKernels); }
inline oa::Span<const oa::ComputeKernel> getUiKernels() { return spanOf(UiKernels); }
inline oa::Span<const oa::ComputeKernel> getAudioKernels() { return spanOf(AudioKernels); }
inline oa::Span<const oa::ComputeKernel> getRenderKernels() { return spanOf(RenderKernels); }
inline oa::Span<const oa::ComputeKernel> getCryptoKernels() { return spanOf(CryptoKernels); }
inline oa::Span<const KernelIdReservation> getReservedKernelIdRanges() {
	return spanOf(ReservedKernelIdRanges);
}

static constexpr oa::Usize getTotalKernelCount() {
	return sizeof(MlKernels) / sizeof(MlKernels[0]) +
	       sizeof(VisionKernels) / sizeof(VisionKernels[0]) +
	       sizeof(UiKernels) / sizeof(UiKernels[0]) +
	       sizeof(AudioKernels) / sizeof(AudioKernels[0]) +
	       sizeof(RenderKernels) / sizeof(RenderKernels[0]) +
	       sizeof(CryptoKernels) / sizeof(CryptoKernels[0]);
}

} // namespace oa::kernelRegistry
