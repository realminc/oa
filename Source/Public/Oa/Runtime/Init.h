#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Runtime/OaVk.h>
#include <array>
#include <cstdint>

// Cooperative-matrix tile shape that the device actually reports as supported
// for a specific (A type, B type, C type, Result type) combination — populated
// at device-init time by enumerating vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
// Do not hardcode 16x16x16 in fused kernels: different GPUs report different shapes
// (Turing: 16x16x16, Ada: 16x8x16, RDNA3: 16x16x32, Xe2: 16x16x16).
// The discovered M/N/K should be threaded into shaders as Slang spec constants.
class OaVkCoopMatShape {
public:
	OaU32  M         = 0;       // Cooperative-matrix rows
	OaU32  N         = 0;       // Cooperative-matrix cols
	OaU32  K         = 0;       // Cooperative-matrix inner
	VkScopeKHR Scope = VK_SCOPE_SUBGROUP_KHR;  // Subgroup or workgroup scope
	OaBool Available = false;   // True if the device reports this combo at all
};

// Per-precision-combo CoopMat shapes discovered at device init.
// Each is the FIRST shape the device reports for that combination
// (matching llama.cpp's "remember the first shape we found" strategy).
class OaVkCoopMatShapes {
public:
	// FP16 input × FP16 input → FP32 accumulator → FP32 result.
	// The most common shape on Turing+/Ada/RDNA3/Xe2.
	OaVkCoopMatShape Fp16AccFp32;

	// FP16 input × FP16 input → FP16 accumulator → FP16 result.
	// Saves shmem + regs on devices that support it.
	OaVkCoopMatShape Fp16AccFp16;

	// BF16 input × BF16 input → FP32 accumulator → FP32 result.
	// What our shipping fused-coopmat-bf16 kernels target.
	OaVkCoopMatShape Bf16AccFp32;

	// INT8 input × INT8 input → INT32 accumulator → INT32 result.
	// For Q4/Q8 quantized matmul (inference, future).
	OaVkCoopMatShape Int8AccInt32;

	// Workgroup-scope variant (larger fragments, NVIDIA-favored).
	// Kept separate from subgroup-scope so routers can pick the portable
	// 16×16×16 path on AMD/Intel and the wider path on NVIDIA when available.
	OaVkCoopMatShape Bf16AccFp32Workgroup;

	// Convenience: was at least one shape with M=N=K=16 + FP32 acc reported?
	// Our kernels are written for this shape today; this stays the "is the
	// hardcoded 16x16x16 path safe?" check until kernels go spec-const.
	OaBool Has16x16x16_Fp32Acc = false;
	OaBool Has16x16x16_Fp16Acc = false;

	// Convenience: total number of shapes the device reported (any combo).
	// Useful for the OA_LOG_COOPMAT_SHAPES diagnostic.
	OaU32  TotalShapesEnumerated = 0;

	// CoopMat2 (NV) capabilities. Populated only when the device advertises
	// VK_NV_cooperative_matrix2 and the feature struct is enabled. These come
	// from VkPhysicalDeviceCooperativeMatrix2FeaturesNV +
	// VkPhysicalDeviceCooperativeMatrix2PropertiesNV. R2 of the gemm-router
	// rewrite consumes them in OaMatmulRegistry::ComputeCapsMask so variants
	// gate on real reported support instead of a single boolean.
	OaBool CoopMat2Supported              = false;
	OaBool CoopMat2WorkgroupScope         = false;
	OaBool CoopMat2FlexibleDimensions     = false;
	OaBool CoopMat2Reductions             = false;
	OaBool CoopMat2PerElementOps          = false;
	OaBool CoopMat2TensorAddressing       = false;
	OaBool CoopMat2BlockLoads             = false;
	OaU32  CoopMat2WorkgroupMaxWgSize     = 0;
	OaU32  CoopMat2FlexibleDimMax         = 0;
	OaU32  CoopMat2WorkgroupReservedSmem  = 0;
};

// Vulkan vendor IDs (per VkPhysicalDeviceProperties::vendorID).
// Used by OaCoopMatTrust + GemmRouter vendor-gating decisions.
inline constexpr OaU32 OaVkVendorIdAmd    = 0x1002U;
inline constexpr OaU32 OaVkVendorIdNvidia = 0x10DEU;
inline constexpr OaU32 OaVkVendorIdIntel  = 0x8086U;
inline constexpr OaU32 OaVkVendorIdArm    = 0x13B5U;

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
[[nodiscard]] bool OaCoopMatTrust(OaU32 InVendorId, OaU32 InDeviceId, OaU32 InDriverId);

// Vendor/driver trust for native BF16 (VK_KHR_shader_bfloat16 shaderBFloat16Type).
// Same reasoning as OaCoopMatTrust — some drivers advertise bf16 but miscompile it.
// Override with OA_FORCE_BF16=1.
[[nodiscard]] bool OaBf16Trust(OaU32 InVendorId, OaU32 InDeviceId, OaU32 InDriverId);

// Render the discovered CoopMat shapes as INFO log lines (one per dtype combo
// + a Has16x16x16 summary). Ungated — caller decides when to emit. Used by:
//   - OaVkComputeDevice::LogCoopMatShapes() — gated by OA_LOG_COOPMAT_SHAPES
//   - OaVkDevice::PrintInfo() — inside the CoreCapabilities block when the
//     same env knob is set.
// Indent each line with the supplied prefix (e.g. "    " to fit under
// "  CoreCapabilities("). Empty if HasCooperativeMatrix is false.
void OaVkLogCoopMatShapes(const OaVkCoopMatShapes& InShapes, const char* InIndent);

// OA Vulkan Extensions & Device Initialization
//
// Canonical extension names, layer names, and minimal specs.
// Internal structures for device creation (documented in oa/Docs/OaVkInit.md).

// Instance layer (enabled when OaVkDevice::Create(enableValidation=true))
inline constexpr const char OaVkLayerKhronosValidation[] = "VK_LAYER_KHRONOS_validation";
inline constexpr std::array<const char*, 1> OaVkInstanceLayerNames{{
	OaVkLayerKhronosValidation,
}};

// Instance extensions (empty; add WSI/surface names here for swapchain)
inline constexpr std::array<const char*, 0> OaVkInstanceExtensionNames{};

// Device extension constants
inline constexpr const char OaVkExtKhrCooperativeMatrix[] = "VK_KHR_cooperative_matrix";
inline constexpr const char OaVkExtNvCooperativeMatrix[] = "VK_NV_cooperative_matrix";
inline constexpr const char OaVkExtKhrPipelineLibrary[] = "VK_KHR_pipeline_library";
inline constexpr const char OaVkExtKhrExternalMemory[] = "VK_KHR_external_memory";
inline constexpr const char OaVkExtKhrExternalMemoryFd[] = "VK_KHR_external_memory_fd";
inline constexpr const char OaVkExtExtExternalMemoryDmaBuf[] = "VK_EXT_external_memory_dma_buf";
inline constexpr const char OaVkExtExtImageDrmFormatModifier[] = "VK_EXT_image_drm_format_modifier";
inline constexpr const char OaVkExtExtQueueFamilyForeign[] = "VK_EXT_queue_family_foreign";
inline constexpr const char OaVkExtKhrShaderBfloat16[] = "VK_KHR_shader_bfloat16";
inline constexpr const char OaVkExtNvCooperativeVector[] = "VK_NV_cooperative_vector";
inline constexpr const char OaVkExtNvCooperativeMatrix2[] = "VK_NV_cooperative_matrix2";
inline constexpr const char OaVkExtNvCooperativeMatrixDecodeVector[] = "VK_NV_cooperative_matrix_decode_vector";
inline constexpr const char OaVkExtKhrSwapchain[] = "VK_KHR_swapchain";
inline constexpr const char OaVkExtKhrMaintenance5[] = "VK_KHR_maintenance5";
inline constexpr const char OaVkExtDeviceGeneratedCommands[] = "VK_EXT_device_generated_commands";
inline constexpr const char OaVkExtKhrVideoQueue[] = "VK_KHR_video_queue";
inline constexpr const char OaVkExtKhrVideoDecodeQueue[] = "VK_KHR_video_decode_queue";
inline constexpr const char OaVkExtKhrVideoDecodeH264[] = "VK_KHR_video_decode_h264";
inline constexpr const char OaVkExtKhrVideoDecodeH265[] = "VK_KHR_video_decode_h265";
inline constexpr const char OaVkExtKhrVideoDecodeAV1[] = "VK_KHR_video_decode_av1";
inline constexpr const char OaVkExtKhrVideoDecodeVP9[] = "VK_KHR_video_decode_vp9";
inline constexpr const char OaVkExtKhrVideoEncodeQueue[] = "VK_KHR_video_encode_queue";
inline constexpr const char OaVkExtKhrVideoEncodeH264[] = "VK_KHR_video_encode_h264";
inline constexpr const char OaVkExtKhrVideoEncodeH265[] = "VK_KHR_video_encode_h265";
inline constexpr const char OaVkExtKhrVideoEncodeAV1[] = "VK_KHR_video_encode_av1";
inline constexpr const char OaVkExtKhrSamplerYcbcr[] = "VK_KHR_sampler_ycbcr_conversion";

// Device extensions to probe from physical device
inline constexpr std::array<const char*, 26> OaVkOptionalDeviceExtensionProbeNames{{
	OaVkExtKhrCooperativeMatrix,
	OaVkExtNvCooperativeMatrix,
	OaVkExtNvCooperativeVector,
	OaVkExtNvCooperativeMatrix2,
	OaVkExtNvCooperativeMatrixDecodeVector,
	OaVkExtKhrPipelineLibrary,
	OaVkExtKhrExternalMemory,
	OaVkExtKhrExternalMemoryFd,
	OaVkExtExtExternalMemoryDmaBuf,
	OaVkExtExtImageDrmFormatModifier,
	OaVkExtExtQueueFamilyForeign,
	OaVkExtKhrShaderBfloat16,
	OaVkExtKhrSwapchain,
	OaVkExtKhrMaintenance5,
	OaVkExtDeviceGeneratedCommands,
	OaVkExtKhrVideoQueue,
	OaVkExtKhrVideoDecodeQueue,
	OaVkExtKhrVideoDecodeH264,
	OaVkExtKhrVideoDecodeH265,
	OaVkExtKhrVideoDecodeAV1,
	OaVkExtKhrVideoDecodeVP9,
	OaVkExtKhrVideoEncodeQueue,
	OaVkExtKhrVideoEncodeH264,
	OaVkExtKhrVideoEncodeH265,
	OaVkExtKhrVideoEncodeAV1,
	OaVkExtKhrSamplerYcbcr,
}};

// Minimal spec constants
inline constexpr const uint32_t OaVkMinApiVersion = VK_API_VERSION_1_3;
inline constexpr const uint32_t OaVkMinComputeQueueSlots = 1;
inline constexpr const uint64_t OaVkMinVramDiscrete = 2ULL * 1024 * 1024 * 1024;
inline constexpr const uint64_t OaVkMinVramIntegrated = 256ULL * 1024 * 1024;

// Compile-time counts
inline constexpr uint32_t OaVkNumInstanceLayers = static_cast<uint32_t>(OaVkInstanceLayerNames.size());
inline constexpr uint32_t OaVkNumInstanceExtensions = static_cast<uint32_t>(OaVkInstanceExtensionNames.size());
inline constexpr uint32_t OaVkNumOptionalDeviceExtensionProbes = static_cast<uint32_t>(OaVkOptionalDeviceExtensionProbeNames.size());

//=============================================================================
// OA Vulkan Internal Structures
//=============================================================================
// Helper structures used during device creation (in Device.cpp).
// Not part of the public API; documented in oa/Docs/OaVkInit.md
//
// See phases in OaVkInit.md:
//   - OaVkPhysExtProbe: Phase 3 (extension probing)
//   - OaVkQueuePlan: Phase 4 (queue planning)
//   - OaVkDeviceFeatureBundle: Phase 5 (feature detection)

struct OaVkPhysExtProbe {
	bool KhrCooperativeMatrix = false;
	bool NvCooperativeMatrix = false;
	bool NvCooperativeVector = false;
	bool NvCooperativeMatrix2 = false;
	bool NvCooperativeMatrixDecodeVector = false;
	bool PipelineLibrary = false;
	bool ExternalMemory = false;
	bool ExternalMemoryFd = false;
	bool ExternalMemoryDmaBuf = false;
	bool ImageDrmFormatModifier = false;
	bool QueueFamilyForeign = false;
	bool KhrShaderBfloat16 = false;
	bool KhrSwapchain = false;
	bool KhrSwapchainMaintenance1 = false;
	bool ExtSwapchainMaintenance1 = false;
	bool KhrMaintenance5 = false;  // Required by VK_EXT_device_generated_commands
	bool ExtDeviceGeneratedCommands = false;  // Phase 2b DGC: GPU-authored compute graph execution
	bool KhrVideoQueue = false;
	bool KhrVideoDecodeQueue = false;
	bool KhrVideoDecodeH264 = false;
	bool KhrVideoDecodeH265 = false;
	bool KhrVideoDecodeAV1 = false;
	bool KhrVideoDecodeVP9 = false;
	bool KhrVideoEncodeQueue = false;
	bool KhrVideoEncodeH264 = false;
	bool KhrVideoEncodeH265 = false;
	bool KhrVideoEncodeAV1 = false;
	bool KhrSamplerYcbcr = false;
	bool KhrBufferDeviceAddress = false;
	bool ExtDescriptorIndexing = false;
	bool KhrTimelineSemaphore = false;
	bool KhrSynchronization2 = false;
	bool KhrDynamicRendering = false;
};

struct OaVkQueuePlan {
	OaU32 ComputeQF = UINT32_MAX;
	OaU32 AsyncComputeQF = UINT32_MAX;
	OaU32 TransferQF = UINT32_MAX;
	OaU32 GraphicsQF = UINT32_MAX;
	OaU32 PresentQF = UINT32_MAX;
	OaU32 VideoDecodeQF = UINT32_MAX;
	OaU32 VideoEncodeQF = UINT32_MAX;
	// Per-family video codec operations queried via VkQueueFamilyVideoPropertiesKHR.
	// Used by decoder Create() to verify the selected video queue supports the
	// target codec (Gap 1 — a family can advertise VIDEO_DECODE but not AV1).
	VkVideoCodecOperationFlagsKHR VideoDecodeCodecOps = 0;
	VkVideoCodecOperationFlagsKHR VideoEncodeCodecOps = 0;
	bool WantsGraphics = false;
	bool ComputeHasMultiQueue = false;
	OaU32 ComputeSlots = 0;
	OaU32 DedicatedTransferSlots = 0;
	bool HasAsync = false;
	OaU32 MainComputeCount = 1;
	OaVec<OaF32> PriorityBacking;
	OaVec<VkDeviceQueueCreateInfo> QueueCIs;
};

struct OaVkDeviceFeatureBundle {
	OaU32 PhysicalApiVersion = VK_API_VERSION_1_0;
	VkPhysicalDeviceFeatures2 SupportedFeatures2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
	};
	VkPhysicalDeviceVulkan11Features Supported11 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
	};
	VkPhysicalDeviceVulkan12Features Supported12 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
	};
	VkPhysicalDeviceVulkan13Features Supported13{};
	VkPhysicalDeviceBufferDeviceAddressFeatures SupportedBufferDeviceAddress{};
	VkPhysicalDeviceDescriptorIndexingFeatures SupportedDescriptorIndexing{};
	VkPhysicalDeviceTimelineSemaphoreFeatures SupportedTimelineSemaphore{};
	VkPhysicalDeviceSynchronization2Features SupportedSynchronization2{};
	VkPhysicalDeviceDynamicRenderingFeatures SupportedDynamicRendering{};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR SupportedBf16{};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR SupportedCoopMat{};
	VkPhysicalDeviceCooperativeVectorFeaturesNV SupportedCoopVec{};
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR SupportedSwapchainMaintenance1{};
#if defined(VK_NV_cooperative_matrix2)
	VkPhysicalDeviceCooperativeMatrix2FeaturesNV SupportedCoopMat2{};
#endif
#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
	VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV SupportedCoopMatDecodeVector{};
#endif

	VkPhysicalDeviceFeatures2 Features2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
	};
	VkPhysicalDeviceVulkan11Features Features11 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
	};
	VkPhysicalDeviceVulkan12Features Features12{};
	VkPhysicalDeviceVulkan13Features Features13 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		.synchronization2 = VK_TRUE,
	};
	VkPhysicalDeviceBufferDeviceAddressFeatures BufferDeviceAddressFeatures{};
	VkPhysicalDeviceDescriptorIndexingFeatures DescriptorIndexingFeatures{};
	VkPhysicalDeviceTimelineSemaphoreFeatures TimelineSemaphoreFeatures{};
	VkPhysicalDeviceSynchronization2Features Synchronization2Features{};
	VkPhysicalDeviceDynamicRenderingFeatures DynamicRenderingFeatures{};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR CoopMatFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
	};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR EnableBf16Feat{};
	VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT DeviceGenFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT,
	};
	VkPhysicalDeviceCooperativeVectorFeaturesNV CoopVecFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
	};
	VkPhysicalDeviceSwapchainMaintenance1FeaturesKHR SwapchainMaintenance1Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_KHR,
		.pNext = nullptr,
		.swapchainMaintenance1 = VK_FALSE,
	};
#if defined(VK_NV_cooperative_matrix2)
	VkPhysicalDeviceCooperativeMatrix2FeaturesNV CoopMat2Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV,
	};
#endif
#if defined(VK_NV_COOPERATIVE_MATRIX_DECODE_VECTOR_EXTENSION_NAME)
	VkPhysicalDeviceCooperativeMatrixDecodeVectorFeaturesNV CoopMatDecodeVectorFeatures = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_DECODE_VECTOR_FEATURES_NV,
	};
#endif
#if defined(VK_KHR_video_decode_vp9)
	// VP9 decode (unlike H.264/H.265/AV1) is gated by an explicit feature bit
	// that must be enabled at device creation, not just the extension.
	VkPhysicalDeviceVideoDecodeVP9FeaturesKHR DecodeVp9Features = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR,
	};
#endif

	bool WantEnableBf16Ext = false;
	VkBool32 CoopMatFeatureKHR = VK_FALSE;
	bool HasCoopMatrix = false;
	bool HasCoopVector = false;
	bool HasCoopMatrix2 = false;
	bool HasCoopMatrixDecodeVector = false;
	bool Has16bit = false;
	bool HasIntDotProduct = false;
	bool HasDeviceGeneratedCommands = false;
	bool HasSwapchainMaintenance1 = false;
	bool HasVideoDecodeVp9 = false;

	// Discovered CoopMat shapes (populated by OaVkRefineCooperativeMatrixCapability
	// in Device.cpp; consumed by Device.cpp::Create to populate Software.CoopMatShapes).
	OaVkCoopMatShapes DiscoveredCoopMatShapes;
};
