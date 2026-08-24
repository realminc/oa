#pragma once

#include <oa/core/types.h>

namespace oa {

// Every shipped compute entry point has a globally unique packed ID
// (namespace + local ordinal). prefix blocks below are a stable serialized contract.
// Dispatch / SPIR-V resolution still keys off Name today; id is stable for checkpoints,
// OAV, tooling, and collision detection as the registry is filled in.

enum class ComputeKernelCategory : oa::U8 {
	None = 0,
	Ml,
	Math,
	Crypto,
	Vision,
	Ui,
	Audio,
	Render,      // Graphics shaders (vertex, fragment, etc.)
	MlUser,
	OtherConsumer,
	TestInternal,
	App,
};

// Packed 64-bit id: high 32 = prefix (block owner), low 32 = local ordinal (unique within block).
// Do not reuse a (prefix, local) pair once merged to main. Do not change prefix for a shipped block.
#define OA_COMPUTE_KERNEL_ID(prefix, local) \
	((static_cast<oa::U64>(static_cast<oa::U32>(prefix)) << 32) | static_cast<oa::U64>(static_cast<oa::U32>(local)))

static constexpr oa::U32 computeKernelIdUnpackPrefix(oa::U64 inPacked) {
	return static_cast<oa::U32>(inPacked >> 32);
}

static constexpr oa::U32 computeKernelIdUnpackLocal(oa::U64 inPacked) {
	return static_cast<oa::U32>(inPacked);
}

static constexpr bool computeKernelIdIsValid(oa::U64 inPacked) {
	return inPacked != 0;
}

// reserved prefix blocks. append new blocks deliberately and never renumber a
// shipped block.
namespace computeKernelPrefix {

static constexpr oa::U32 Unassigned = 0;
static constexpr oa::U32 TestInternal = 1; // local 1.. — dev/CI only, not for .oam
static constexpr oa::U32 Ml = 0x0008'1000; // OA Core/ML compute shaders
static constexpr oa::U32 Crypto = 0x0000'0200;
static constexpr oa::U32 Vision = 0x0008'2000; // oa lib Vision/image/video shaders
static constexpr oa::U32 Ui = 0x0008'3000;     // oa lib UI/presentation shaders
static constexpr oa::U32 Audio = 0x0008'4000;  // oa lib Audio/DSP shaders
static constexpr oa::U32 Render = 0x0008'5000; // oa lib Render graphics shaders (vertex/fragment)
static constexpr oa::U32 MlUser = 0x0000'1000; // ml repo architecture shaders
static constexpr oa::U32 Chain = 0x0000'2000;
static constexpr oa::U32 App = 0x0000'3000; // oa apps (modelctl, etc.)

} // namespace computeKernelPrefix

// Stable names for fixed kernels. Each include is emitted by the authority
// that owns the corresponding kernel family.
namespace computeKernelId {
#include <oa/runtime/gen/kernelIdsStandalone.inl>
#include <oa/runtime/gen/tileComputeKernel.h>
#include <oa/runtime/gen/computeKernelIds.inl>

} // namespace computeKernelId

// Metadata row for docs, generators, and future validation (a stable type id + kind, like a node-type registry).
class ComputeKernel {
public:
	const char* name; // SPIR-V registry / dispatch string — unique across the process
	oa::U64 id;         // OA_COMPUTE_KERNEL_ID(prefix, local); 0 = not yet registered
	ComputeKernelCategory category;
	const char* origin; // short source tag: "oa", "ml", "chain"
};

// Composed generated rows for shipped OA SPIR-V names (see the canonical
// registry document).

[[nodiscard]] oa::Span<const ComputeKernel> computeKernelRegistrySpan();

[[nodiscard]] const ComputeKernel* computeKernelFindByPackedId(oa::U64 inPackedId);

[[nodiscard]] const ComputeKernel* computeKernelFindByName(const char* inName);

// True for kernels that use OA's default compute bindless pipeline layout
// (set=0, binding=0 RWByteAddressBuffer heap[]). False for kernels that need
// an image/presentation-specific pipeline layout and must not be loaded via
// oavk::Dispatch/oa::PipelineRegistry.
[[nodiscard]] bool computeKernelUsesDefaultBindlessPipeline(const char* inName);

} // namespace oa
