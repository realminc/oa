#pragma once

#include <Oa/Core/Types.h>

// Every shipped compute entry point has a globally unique packed ID
// (namespace + local ordinal). Prefix blocks below are a stable serialized contract.
// Dispatch / SPIR-V resolution still keys off Name today; Id is stable for checkpoints,
// OAV, tooling, and collision detection as the registry is filled in.

enum class OaComputeKernelCategory : OaU8 {
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

using OaKernelId = OaU64;

// Packed 64-bit id: high 32 = prefix (block owner), low 32 = local ordinal (unique within block).
// Do not reuse a (Prefix, Local) pair once merged to main. Do not change Prefix for a shipped block.
#define OA_COMPUTE_KERNEL_ID(Prefix, Local) \
	((static_cast<OaU64>(static_cast<OaU32>(Prefix)) << 32) | static_cast<OaU64>(static_cast<OaU32>(Local)))

static constexpr OaU32 OaComputeKernelIdUnpackPrefix(OaU64 InPacked) {
	return static_cast<OaU32>(InPacked >> 32);
}

static constexpr OaU32 OaComputeKernelIdUnpackLocal(OaU64 InPacked) {
	return static_cast<OaU32>(InPacked);
}

static constexpr bool OaComputeKernelIdIsValid(OaU64 InPacked) {
	return InPacked != 0;
}

// Reserved prefix blocks. Append new blocks deliberately and never renumber a
// shipped block.
namespace OaComputeKernelPrefix {

static constexpr OaU32 Unassigned = 0;
static constexpr OaU32 TestInternal = 1; // Local 1.. — dev/CI only, not for .oam
static constexpr OaU32 Ml = 0x0008'1000; // oa lib ML shaders (CMake OA_ML_SHADER_REG order)
static constexpr OaU32 Crypto = 0x0000'0200;
static constexpr OaU32 Vision = 0x0008'2000; // oa lib Vision/image/video shaders
static constexpr OaU32 Ui = 0x0008'3000;     // oa lib UI/presentation shaders
static constexpr OaU32 Audio = 0x0008'4000;  // oa lib Audio/DSP shaders
static constexpr OaU32 Render = 0x0008'5000; // oa lib Render graphics shaders (vertex/fragment)
static constexpr OaU32 MlUser = 0x0000'1000; // ml repo architecture shaders
static constexpr OaU32 Chain = 0x0000'2000;
static constexpr OaU32 App = 0x0000'3000; // oa apps (modelctl, etc.)

} // namespace OaComputeKernelPrefix

// Common foundation ids (locals match KernelRegistry.h and the canonical registry doc).
namespace OaComputeKernelId {

static constexpr OaKernelId Silu = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 23);
static constexpr OaKernelId Relu = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 27);
static constexpr OaKernelId Softmax = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 33);
static constexpr OaKernelId CrossEntropy = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 35);
static constexpr OaKernelId CrossEntropyBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 36);
static constexpr OaKernelId CrossEntropyLossGradBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 121);
static constexpr OaKernelId LinearWeightBiasBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 123);
static constexpr OaKernelId LinearDataReluBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 125);
static constexpr OaKernelId Sgd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 48);
static constexpr OaKernelId Adamw = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 50);
static constexpr OaKernelId SgdMomentum = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 51);
static constexpr OaKernelId AdamwMany4 = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 189);
static constexpr OaKernelId SmoothL1 = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 190);
static constexpr OaKernelId SmoothL1Bwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 191);
static constexpr OaKernelId Mse = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 192);
static constexpr OaKernelId MseBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 193);
static constexpr OaKernelId L1 = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 194);
static constexpr OaKernelId L1Bwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 195);
static constexpr OaKernelId Bce = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 196);
static constexpr OaKernelId BceBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 197);
static constexpr OaKernelId Scale = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 7);
static constexpr OaKernelId Fill = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 6);
static constexpr OaKernelId Sum = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 11);
static constexpr OaKernelId ReduceCols = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 14);
static constexpr OaKernelId Transpose = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 10);
static constexpr OaKernelId GemmNaive = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 72);
#include <Oa/Runtime/OaTileComputeKernel.gen.h>
// IDs 126,127,132-135,144-154,184,212-231 retired (CoopMat2 NV kernels removed).
// ID 124 retired: was GemmBiasCoopMatBf16 (superseded by GemmBiasBf16).
static constexpr OaKernelId CvtNv12ToRgb = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 2);
static constexpr OaKernelId CvtNv12YcbcrToRgba = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Vision, 4);
static constexpr OaKernelId BlitImageRgba = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 3);
static constexpr OaKernelId DrawRectOutline = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 4);
static constexpr OaKernelId DrawRectOutlines = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 5);
static constexpr OaKernelId DrawGlyphs = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 6);
static constexpr OaKernelId DrawRect = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 7);
static constexpr OaKernelId DrawWaveform = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ui, 8);
static constexpr OaKernelId AudioWaveformEnvelope = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Audio, 7);
static constexpr OaKernelId RlGae = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 264);
static constexpr OaKernelId RlPpoClip = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 265);
static constexpr OaKernelId RlPpoClipBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 266);
static constexpr OaKernelId RlRolloutAppend = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 272);
static constexpr OaKernelId RlRolloutReset = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 273);
#include <Oa/Runtime/ComputeKernelIds.gen.inl>
static constexpr OaKernelId LogSoftmax = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 270);
static constexpr OaKernelId LogSoftmaxBwd = OA_COMPUTE_KERNEL_ID(OaComputeKernelPrefix::Ml, 271);

} // namespace OaComputeKernelId

// Metadata row for docs, generators, and future validation (a stable type id + kind, like a node-type registry).
class OaComputeKernel {
public:
	const char* Name; // SPIR-V registry / dispatch string — unique across the process
	OaU64 Id;         // OA_COMPUTE_KERNEL_ID(Prefix, Local); 0 = not yet registered
	OaComputeKernelCategory Category;
	const char* Origin; // short source tag: "oa", "ml", "chain"
};

// Authoritative rows for shipped OA SPIR-V names (see the canonical registry doc).
// ML row order matches CMake OA_ML_SHADER_REG + NaiveMatmul + MatmulCoopMat.

[[nodiscard]] OaSpan<const OaComputeKernel> OaComputeKernelRegistrySpan();

[[nodiscard]] const OaComputeKernel* OaComputeKernelFindByPackedId(OaU64 InPackedId);

[[nodiscard]] const OaComputeKernel* OaComputeKernelFindByName(const char* InName);

// True for kernels that use OA's default compute bindless pipeline layout
// (set=0, binding=0 RWByteAddressBuffer heap[]). False for kernels that need
// an image/presentation-specific pipeline layout and must not be loaded via
// OaVkDispatch/OaPipelineRegistry.
[[nodiscard]] bool OaComputeKernelUsesDefaultBindlessPipeline(const char* InName);
