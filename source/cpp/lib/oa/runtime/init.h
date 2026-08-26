#pragma once

#include <oa/core/types.h>
#include <oa/core/std/array.h>
#include <oa/runtime/oaVk.h>

namespace oavk {

// Cooperative-matrix tile shape that the device actually reports as supported
// for a specific (A type, B type, C type, result type) combination — populated
// at device-init time by enumerating vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
// Do not hardcode 16x16x16 in fused kernels: different GPUs report different shapes
// (Turing: 16x16x16, Ada: 16x8x16, RDNA3: 16x16x32, Xe2: 16x16x16).
// The discovered M/N/K should be threaded into shaders as slang spec constants.
class CoopMatShape {
public:
	oa::U32  m         = 0;       // Cooperative-matrix rows
	oa::U32  n         = 0;       // Cooperative-matrix cols
	oa::U32  k         = 0;       // Cooperative-matrix inner
	VkScopeKHR scope = VK_SCOPE_SUBGROUP_KHR;  // Subgroup or workgroup scope
	oa::Bool available = false;   // True if the device reports this combo at all
};

// Per-precision-combo CoopMat shapes discovered at device init.
// Each is the FIRST shape the device reports for that combination
// (matching llama.cpp's "remember the first shape we found" strategy).
class CoopMatShapes {
public:
	// FP16 input × FP16 input → FP32 accumulator → FP32 result.
	// The most common shape on Turing+/Ada/RDNA3/Xe2.
	CoopMatShape fp16AccFp32;

	// FP16 input × FP16 input → FP16 accumulator → FP16 result.
	// Saves shmem + regs on devices that support it.
	CoopMatShape fp16AccFp16;

	// BF16 input × BF16 input → FP32 accumulator → FP32 result.
	// what our shipping fused-coopmat-bf16 kernels target.
	CoopMatShape bf16AccFp32;

	// INT8 input × INT8 input → INT32 accumulator → INT32 result.
	// reserved for a future device-qualified integer-dot/CoopMat Q4/Q8 route.
	// The shipped portable fused Q4/Q8 path accumulates in Float32 and does not
	// require this capability.
	CoopMatShape int8AccInt32;

	// workgroup-scope variant (larger fragments, NVIDIA-favored).
	// Kept separate from subgroup-scope so routers can pick the portable
	// 16×16×16 path on AMD/Intel and the wider path on NVIDIA when available.
	CoopMatShape bf16AccFp32Workgroup;

	// Convenience: was at least one shape with M=N=K=16 + FP32 acc reported?
	// Our kernels are written for this shape today; this stays the "is the
	// hardcoded 16x16x16 path safe?" check until kernels go spec-const.
	oa::Bool has16x16x16_Fp32Acc = false;
	oa::Bool has16x16x16_Fp16Acc = false;

	// Convenience: total number of shapes the device reported (any combo).
	// Useful for the OA_LOG_COOPMAT_SHAPES diagnostic.
	oa::U32  totalShapesEnumerated = 0;

	// coopMat2 (NV) capabilities. Populated only when the device advertises
	// VK_NV_cooperative_matrix2 and the feature struct is enabled. These come
	// from VkPhysicalDeviceCooperativeMatrix2FeaturesNV +
	// VkPhysicalDeviceCooperativeMatrix2PropertiesNV. R2 of the gemm-router
	// rewrite consumes them in oa::matmulRegistry::ComputeCapsMask so variants
	// gate on real reported support instead of a single boolean.
	oa::Bool coopMat2Supported              = false;
	oa::Bool coopMat2WorkgroupScope         = false;
	oa::Bool coopMat2FlexibleDimensions     = false;
	oa::Bool coopMat2Reductions             = false;
	oa::Bool coopMat2PerElementOps          = false;
	oa::Bool coopMat2TensorAddressing       = false;
	oa::Bool coopMat2BlockLoads             = false;
	oa::U32  coopMat2WorkgroupMaxWgSize     = 0;
	oa::U32  coopMat2FlexibleDimMax         = 0;
	oa::U32  coopMat2WorkgroupReservedSmem  = 0;
};

// vulkan vendor iDs (per VkPhysicalDeviceProperties::vendorID).
// Used by coopMatTrust + GemmRouter vendor-gating decisions.
inline constexpr oa::U32 VendorIdAmd    = 0x1002U;
inline constexpr oa::U32 VendorIdNvidia = 0x10DEU;
inline constexpr oa::U32 VendorIdIntel  = 0x8086U;
inline constexpr oa::U32 VendorIdArm    = 0x13B5U;

// Vendor-trust decision for KHR_cooperative_matrix.
//
// Returns true iff the (vendor, deviceId, driverId) combination is known to
// implement CoopMat correctly. Mirrors llama.cpp's vendor blacklist:
//   - AMD proprietary/open-source blob: reports support on all GPUs but
//     miscompiles or crashes pre-RDNA3 (driverId 1 = AMD_PROPRIETARY,
//     driverId 2 = AMD_OPEN_SOURCE). RADV (driverId 3) is trustworthy.
//   - Intel pre-Xe2: reports support but regresses below scalar FP32.
//   - NVIDIA Turing+ and other vendors: trusted by default (CoopMat
//     non-presence on pre-Turing already gates that case).
//
// Override at runtime with OA_FORCE_COOPMAT=1 to bypass the trust check
// (testing on new hardware, CI on emulators, etc.).
[[nodiscard]] bool coopMatTrust(oa::U32 inVendorId, oa::U32 inDeviceId, oa::U32 inDriverId);

// Vendor/driver trust for native BF16 (VK_KHR_shader_bfloat16 shaderBFloat16Type).
// Same reasoning as coopMatTrust — some drivers advertise bf16 but miscompile it.
// Override with OA_FORCE_BF16=1.
[[nodiscard]] bool bf16Trust(oa::U32 inVendorId, oa::U32 inDeviceId, oa::U32 inDriverId);

// Render the discovered CoopMat shapes as INFO log lines (one per dtype combo
// + a Has16x16x16 summary). Ungated — caller decides when to emit. Used by:
//   - oavk::ComputeDevice::logCoopMatShapes() — gated by OA_LOG_COOPMAT_SHAPES
//   - oavk::Device::printInfo() — inside the CoreCapabilities block when the
//     same env knob is set.
// indent each line with the supplied prefix (e.g. "    " to fit under
// "  coreCapabilities("). Empty if hasCooperativeMatrix is false.
void logCoopMatShapes(const CoopMatShapes& inShapes, const char* inIndent);

// OA vulkan Extensions & Device Initialization
//
// Canonical extension names, layer names, minimal specs, and internal device
// creation structures.

// instance layer (enabled when oavk::Device::create(enableValidation=true))
inline constexpr const char LayerKhronosValidation[] = "VK_LAYER_KHRONOS_validation";
inline constexpr oa::Array<const char*, 1> InstanceLayerNames{
	LayerKhronosValidation,
};

// instance extensions (empty; add WSI/surface names here for swapchain)
inline constexpr oa::Array<const char*, 0> InstanceExtensionNames{};

// Device extension constants
inline constexpr const char ExtKhrCooperativeMatrix[] = "VK_KHR_cooperative_matrix";
inline constexpr const char ExtNvCooperativeMatrix[] = "VK_NV_cooperative_matrix";
inline constexpr const char ExtKhrPipelineLibrary[] = "VK_KHR_pipeline_library";
inline constexpr const char ExtKhrExternalMemory[] = "VK_KHR_external_memory";
inline constexpr const char ExtKhrExternalMemoryFd[] = "VK_KHR_external_memory_fd";
inline constexpr const char ExtExtExternalMemoryDmaBuf[] = "VK_EXT_external_memory_dma_buf";
inline constexpr const char ExtExtImageDrmFormatModifier[] = "VK_EXT_image_drm_format_modifier";
inline constexpr const char ExtExtQueueFamilyForeign[] = "VK_EXT_queue_family_foreign";
inline constexpr const char ExtKhrShaderBfloat16[] = "VK_KHR_shader_bfloat16";
inline constexpr const char ExtNvCooperativeVector[] = "VK_NV_cooperative_vector";
inline constexpr const char ExtNvCooperativeMatrix2[] = "VK_NV_cooperative_matrix2";
inline constexpr const char ExtNvCooperativeMatrixDecodeVector[] = "VK_NV_cooperative_matrix_decode_vector";
inline constexpr const char ExtKhrSwapchain[] = "VK_KHR_swapchain";
inline constexpr const char ExtKhrMaintenance5[] = "VK_KHR_maintenance5";
inline constexpr const char ExtDeviceGeneratedCommands[] = "VK_EXT_device_generated_commands";
inline constexpr const char ExtKhrVideoQueue[] = "VK_KHR_video_queue";
inline constexpr const char ExtKhrVideoDecodeQueue[] = "VK_KHR_video_decode_queue";
inline constexpr const char ExtKhrVideoDecodeH264[] = "VK_KHR_video_decode_h264";
inline constexpr const char ExtKhrVideoDecodeH265[] = "VK_KHR_video_decode_h265";
inline constexpr const char ExtKhrVideoDecodeAV1[] = "VK_KHR_video_decode_av1";
inline constexpr const char ExtKhrVideoDecodeVP9[] = "VK_KHR_video_decode_vp9";
inline constexpr const char ExtKhrVideoEncodeQueue[] = "VK_KHR_video_encode_queue";
inline constexpr const char ExtKhrVideoEncodeH264[] = "VK_KHR_video_encode_h264";
inline constexpr const char ExtKhrVideoEncodeH265[] = "VK_KHR_video_encode_h265";
inline constexpr const char ExtKhrVideoEncodeAV1[] = "VK_KHR_video_encode_av1";
inline constexpr const char ExtKhrSamplerYcbcr[] = "VK_KHR_sampler_ycbcr_conversion";
inline constexpr const char ExtKhrCalibratedTimestamps[] = "VK_KHR_calibrated_timestamps";
inline constexpr const char ExtExtCalibratedTimestamps[] = "VK_EXT_calibrated_timestamps";

// Device extensions to probe from physical device
inline constexpr oa::Array<const char*, 28> OptionalDeviceExtensionProbeNames{
	ExtKhrCooperativeMatrix,
	ExtNvCooperativeMatrix,
	ExtNvCooperativeVector,
	ExtNvCooperativeMatrix2,
	ExtNvCooperativeMatrixDecodeVector,
	ExtKhrPipelineLibrary,
	ExtKhrExternalMemory,
	ExtKhrExternalMemoryFd,
	ExtExtExternalMemoryDmaBuf,
	ExtExtImageDrmFormatModifier,
	ExtExtQueueFamilyForeign,
	ExtKhrShaderBfloat16,
	ExtKhrSwapchain,
	ExtKhrMaintenance5,
	ExtDeviceGeneratedCommands,
	ExtKhrVideoQueue,
	ExtKhrVideoDecodeQueue,
	ExtKhrVideoDecodeH264,
	ExtKhrVideoDecodeH265,
	ExtKhrVideoDecodeAV1,
	ExtKhrVideoDecodeVP9,
	ExtKhrVideoEncodeQueue,
	ExtKhrVideoEncodeH264,
	ExtKhrVideoEncodeH265,
	ExtKhrVideoEncodeAV1,
	ExtKhrSamplerYcbcr,
	ExtKhrCalibratedTimestamps,
	ExtExtCalibratedTimestamps,
};

// Minimal spec constants
inline constexpr const uint32_t MinApiVersion = VK_API_VERSION_1_3;
inline constexpr const uint32_t MinComputeQueueSlots = 1;
inline constexpr const uint64_t MinVramDiscrete = 2ULL * 1024 * 1024 * 1024;
inline constexpr const uint64_t MinVramIntegrated = 256ULL * 1024 * 1024;

// compile-time counts
inline constexpr uint32_t NumInstanceLayers = static_cast<uint32_t>(InstanceLayerNames.size());
inline constexpr uint32_t NumInstanceExtensions = static_cast<uint32_t>(InstanceExtensionNames.size());
inline constexpr uint32_t NumOptionalDeviceExtensionProbes = static_cast<uint32_t>(OptionalDeviceExtensionProbeNames.size());

//=============================================================================
// OA vulkan Internal Structures
//=============================================================================
// Helper structures used during device creation (in Device.cpp).
// Not part of the public API.

struct PhysicalExtensionProbe {
	bool khrCooperativeMatrix = false;
	bool nvCooperativeMatrix = false;
	bool nvCooperativeVector = false;
	bool nvCooperativeMatrix2 = false;
	bool nvCooperativeMatrixDecodeVector = false;
	bool pipelineLibrary = false;
	bool externalMemory = false;
	bool externalMemoryFd = false;
	bool externalMemoryDmaBuf = false;
	bool imageDrmFormatModifier = false;
	bool queueFamilyForeign = false;
	bool khrShaderBfloat16 = false;
	bool khrSwapchain = false;
	bool khrSwapchainMaintenance1 = false;
	bool extSwapchainMaintenance1 = false;
	bool khrMaintenance5 = false;  // Required by VK_EXT_device_generated_commands
	bool extDeviceGeneratedCommands = false;  // phase 2b DGC: GPU-authored compute graph execution
	bool khrVideoQueue = false;
	bool khrVideoDecodeQueue = false;
	bool khrVideoDecodeH264 = false;
	bool khrVideoDecodeH265 = false;
	bool khrVideoDecodeAV1 = false;
	bool khrVideoDecodeVP9 = false;
	bool khrVideoEncodeQueue = false;
	bool khrVideoEncodeH264 = false;
	bool khrVideoEncodeH265 = false;
	bool khrVideoEncodeAV1 = false;
	bool khrSamplerYcbcr = false;
	bool khrBufferDeviceAddress = false;
	bool extDescriptorIndexing = false;
	bool khrTimelineSemaphore = false;
	bool khrSynchronization2 = false;
	bool khrDynamicRendering = false;
	bool khrCalibratedTimestamps = false;
	bool extCalibratedTimestamps = false;
};

struct QueuePlan {
	oa::U32 computeQF = UINT32_MAX;
	oa::U32 asyncComputeQF = UINT32_MAX;
	oa::U32 transferQF = UINT32_MAX;
	oa::U32 graphicsQF = UINT32_MAX;
	oa::U32 presentQF = UINT32_MAX;
	oa::U32 videoDecodeQF = UINT32_MAX;
	oa::U32 videoEncodeQF = UINT32_MAX;
	// Per-family video codec operations queried via VkQueueFamilyVideoPropertiesKHR.
	// Used by decoder create() to verify the selected video queue supports the
	// target codec (gap 1 — a family can advertise VIDEO_DECODE but not AV1).
	VkVideoCodecOperationFlagsKHR videoDecodeCodecOps = 0;
	VkVideoCodecOperationFlagsKHR videoEncodeCodecOps = 0;
	bool wantsGraphics = false;
	bool computeHasMultiQueue = false;
	oa::U32 computeSlots = 0;
	oa::U32 dedicatedTransferSlots = 0;
	bool hasAsync = false;
	oa::U32 mainComputeCount = 1;
	oa::Vec<oa::F32> priorityBacking;
	oa::Vec<VkDeviceQueueCreateInfo> queueCIs;
};

struct DeviceFeatureBundle {
	oa::U32 physicalApiVersion = VK_API_VERSION_1_0;
	VkPhysicalDeviceFeatures2 supportedFeatures2{};
	VkPhysicalDeviceVulkan11Features supported11{};
	VkPhysicalDeviceVulkan12Features supported12{};
	VkPhysicalDeviceVulkan13Features supported13{};
	VkPhysicalDeviceBufferDeviceAddressFeatures supportedBufferDeviceAddress{};
	VkPhysicalDeviceDescriptorIndexingFeatures supportedDescriptorIndexing{};
	VkPhysicalDeviceTimelineSemaphoreFeatures supportedTimelineSemaphore{};
	VkPhysicalDeviceSynchronization2Features supportedSynchronization2{};
	VkPhysicalDeviceDynamicRenderingFeatures supportedDynamicRendering{};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR supportedBf16{};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR supportedCoopMat{};
	VkPhysicalDeviceCooperativeVectorFeaturesNV supportedCoopVec{};
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR supportedSwapchainMaintenance1{};
#if defined(VK_NV_cooperative_matrix2)
	VkPhysicalDeviceCooperativeMatrix2FeaturesNV supportedCoopMat2{};
#endif
#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
	VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV supportedCoopMatDecodeVector{};
#endif

	VkPhysicalDeviceFeatures2 features2{};
	VkPhysicalDeviceVulkan11Features features11{};
	VkPhysicalDeviceVulkan12Features features12{};
	VkPhysicalDeviceVulkan13Features features13{};
	VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
	VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{};
	VkPhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
	VkPhysicalDeviceSynchronization2Features synchronization2Features{};
	VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures{};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR enableBf16Feat{};
	VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT deviceGenFeatures{};
	VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeatures{};
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR swapchainMaintenance1Features{};
#if defined(VK_NV_cooperative_matrix2)
	VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopMat2Features{};
#endif
#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
	VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV coopMatDecodeVectorFeatures{};
#endif
#if defined(VK_KHR_video_decode_vp9)
	// VP9 decode (unlike H.264/H.265/AV1) is gated by an explicit feature bit
	// that must be enabled at device creation, not just the extension.
	VkPhysicalDeviceVideoDecodeVP9FeaturesKHR decodeVp9Features{};
#endif

	bool wantEnableBf16Ext = false;
	VkBool32 CoopMatFeatureKHR = VK_FALSE;
	bool hasCoopMatrix = false;
	bool hasCoopVector = false;
	bool hasCoopMatrix2 = false;
	bool hasCoopMatrixDecodeVector = false;
	bool has16bit = false;
	bool hasIntDotProduct = false;
	bool hasDeviceGeneratedCommands = false;
	bool hasSwapchainMaintenance1 = false;
	bool hasVideoDecodeVp9 = false;

	// Discovered CoopMat shapes (populated by cooperative-matrix feature discovery
	// in Device.cpp; consumed by Device.cpp::Create to populate software.coopMatShapes).
	CoopMatShapes discoveredCoopMatShapes;
};

} // namespace oavk
