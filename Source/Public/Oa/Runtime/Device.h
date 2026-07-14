// OA Vulkan — logical device, queues, capability probe.
// Physical enumeration pick score / estimates / survey log: Oa/Runtime/Rate.h (included below).
// OaVkDeviceInfo groups OaVkDeviceHardwareInfo (PCI, VRAM, limits, heuristics) and
// OaVkDeviceSoftwareInfo (driver, API, enabled extensions, shader / extension caps).
#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Device.h>
#include <Oa/Runtime/OaVk.h>
#include <Oa/Runtime/Init.h>     // OaVkCoopMatShape, OaVkCoopMatShapes
#include <Oa/Runtime/Rate.h>

#include <limits>

inline constexpr OaU32 OaVkEnumerationIndexUnset = std::numeric_limits<OaU32>::max();
inline constexpr OaU32 OaVkPciDeviceIdUnknown    = std::numeric_limits<OaU32>::max();


class OaVkQueues {
public:
	void* ComputeQueue            = nullptr;
	void* AsyncComputeQueue       = nullptr;
	void* TransferQueue           = nullptr;
	void* GraphicsQueue           = nullptr;
	void* PresentQueue            = nullptr;
	void* VideoDecodeQueue        = nullptr;  // VK_KHR_video_decode_queue
	void* VideoEncodeQueue        = nullptr;  // VK_KHR_video_encode_queue
	OaU32 ComputeQueueFamily      = 0;
	OaU32 AsyncComputeQueueFamily = 0;
	OaU32 TransferQueueFamily     = 0;
	OaU32 GraphicsQueueFamily     = OaVkEnumerationIndexUnset;
	OaU32 PresentQueueFamily      = OaVkEnumerationIndexUnset;
	OaU32 VideoDecodeQueueFamily  = OaVkEnumerationIndexUnset;
	OaU32 VideoEncodeQueueFamily  = OaVkEnumerationIndexUnset;
	OaBool HasAsyncCompute        = false;
	OaBool HasPresentation        = false;
	OaBool HasVideoDecodeQueue    = false;
	OaBool HasVideoEncodeQueue    = false;
	// Per-family video codec operations (from VkQueueFamilyVideoPropertiesKHR).
	// Decoder Create() checks these to verify the queue supports the target codec.
	VkVideoCodecOperationFlagsKHR VideoDecodeCodecOps = 0;
	VkVideoCodecOperationFlagsKHR VideoEncodeCodecOps = 0;
	OaU32 ComputeQueueSlotCount           = 0;
	OaU32 DedicatedTransferQueueSlotCount = 0;
};


class OaVkDeviceHardwareInfo {
public:
	OaString     DeviceName;
	OaString     VendorName;
	OaU32        VendorId                       = 0;
	OaU32        DeviceId                       = 0;
	OaDeviceType DeviceType                     = OaDeviceType::Host;
	OaU64        VramBytes                      = 0;
	OaU32        SubgroupSize                   = 0;
	OaU32        MaxComputeWorkGroupSize        = 0;
	OaU32        MaxComputeWorkGroupInvocations = 0;
	OaU32        NumSMs                         = 0;  // Streaming multiprocessors (NVIDIA) / CUs (AMD)
	OaU32        MaxPerStageDescriptorUpdateAfterBindStorageBuffers = 0;
	OaU32        MaxPerStageDescriptorUpdateAfterBindSampledImages = 0;
	OaU32        MaxPerStageDescriptorUpdateAfterBindSamplers = 0;
	OaBool       HasSAM                         = false;
	OaU64        PickRating                     = 0;
	OaU32        EnumerationIndex               = OaVkEnumerationIndexUnset;
	OaF64        EstMemBandwidthGbps            = 0.0;
	OaF64        EstPeakTflopsF32               = 0.0;
};


class OaVkDeviceSoftwareInfo {
public:
	OaString         DriverVersion;
	OaString         ApiVersion;
	// Numeric physical-device API version. Runtime feature adapters may expose
	// newer behavior through extensions, but consumers such as VMA must still
	// receive the real core version advertised by the device.
	OaU32            ApiVersionPacked = VK_API_VERSION_1_0;
	OaU32            DriverId = 0;
	OaString         DriverName;
	OaString         DriverInfo;
	OaVec<OaString>  EnabledDeviceExtensions;
	
	OaBool HasCooperativeMatrix                    = false;
	OaBool HasCooperativeVector                    = false;  // VK_NV_cooperative_vector (Blackwell+)
	OaBool HasCooperativeMatrix2                   = false;  // VK_NV_cooperative_matrix2
	OaBool HasCooperativeMatrixDecodeVector        = false;  // VK_NV_cooperative_matrix_decode_vector
	// Discovered CoopMat shapes from vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
	// Populated when HasCooperativeMatrix is true. Future PR will thread these as Slang
	// spec constants into fused-CoopMat kernels (replacing today's hardcoded 16x16x16).
	OaVkCoopMatShapes CoopMatShapes;
	OaBool HasPipelineLibrary                      = false;
	OaBool Has16BitStorage                         = false;
	OaBool ShaderFloat16Enabled                    = false;
	OaBool ShaderBfloat16ExtensionEnabled          = false;
	OaBool ShaderBfloat16TypeEnabled               = false;
	OaBool ShaderBfloat16DotProductEnabled         = false;
	OaBool ShaderBfloat16CooperativeMatrixEnabled  = false;
	OaBool ShaderIntegerDotProductEnabled          = false;  // VK_KHR_shader_integer_dot_product (INT8 quantization)
	OaBool HasExternalMemoryFd                     = false;
	
	// Device-generated commands (Phase 2b DGC)
	OaBool HasDeviceGeneratedCommands              = false;  // VK_EXT_device_generated_commands (GPU-authored execution)
	
	// Vulkan Video extensions (Phase 2.1)
	OaBool HasVideoQueue                           = false;  // VK_KHR_video_queue
	OaBool HasVideoDecodeQueue                     = false;  // VK_KHR_video_decode_queue
	OaBool HasVideoDecodeH264                      = false;  // VK_KHR_video_decode_h264
	OaBool HasVideoDecodeH265                      = false;  // VK_KHR_video_decode_h265
	OaBool HasVideoDecodeAV1                       = false;  // VK_KHR_video_decode_av1
	OaBool HasVideoDecodeVP9                     = false;  // VK_KHR_video_decode_vp9
	OaBool HasVideoEncodeQueue                     = false;  // VK_KHR_video_encode_queue
	OaBool HasVideoEncodeH264                      = false;  // VK_KHR_video_encode_h264
	OaBool HasVideoEncodeH265                      = false;  // VK_KHR_video_encode_h265
	OaBool HasVideoEncodeAV1                       = false;  // VK_KHR_video_encode_av1
	OaBool HasSamplerYcbcrConversion               = false;  // VK_KHR_sampler_ycbcr_conversion
};


class OaVkDeviceInfo {
public:
	OaVkDeviceHardwareInfo Hardware;
	OaVkDeviceSoftwareInfo Software;
};


class OaVkDevice : public OaDevice {
public:
	// Vulkan implementation of a logical OaDevice: base holds Type + mesh Index.

	// Data
	void* Instance       = nullptr;
	void* PhysicalDevice = nullptr;
	void* Device         = nullptr;
	OaBool OwnsInstance  = true;
	OaVkQueues    Queues;
	OaVkDeviceInfo Info;

	// -----------------------------------------------------------------------
	// Create — single physical device.
	//
	// InHintNeedsPresentation:
	//   Pass true when the caller intends to attach a VkSurfaceKHR later (e.g.
	//   via OaGraphicsEngine::InitPresentation).  The device will be created
	//   with VK_QUEUE_GRAPHICS_BIT + VK_KHR_swapchain without needing the
	//   surface at creation time, so SDL_Vulkan_CreateSurface can be called
	//   against the resulting Instance handle.
	// -----------------------------------------------------------------------
	[[nodiscard]] static OaResult<OaVkDevice> Create(
		OaStringView              InAppName,
		OaBool                    InEnableValidation,
		OaDeviceType              InPreferred               = OaDeviceType::VkDiscrete,
		OaU32                     InForceEnumerationIndex   = OaVkEnumerationIndexUnset,
		OaU32                     InAppVersionPatch         = 1,
		OaSpan<const char* const> InInstanceExtraExtensions = {},
		OaBool                    InHintNeedsPresentation   = false
	);

	// -----------------------------------------------------------------------
	// CreateFromPhysical — re-use an existing VkInstance.
	//
	// InSurface (VkSurfaceKHR):
	//   When non-null, queue planning will pick a family that supports
	//   graphics + present for this surface and enable VK_KHR_swapchain.
	//
	// InHintNeedsPresentation:
	//   When true and InSurface is null, a graphics-capable queue family and
	//   VK_KHR_swapchain are still enabled so the device is ready for
	//   swapchain creation once a surface becomes available.
	// -----------------------------------------------------------------------
	[[nodiscard]] static OaResult<OaVkDevice> CreateFromPhysical(
		void*  InInstance,
		void*  InPhysicalDevice,
		OaBool InEnableValidation,
		OaU64  InPickRating            = 0,
		OaU32  InEnumerationIndex      = OaVkEnumerationIndexUnset,
		void*  InSurface               = nullptr,
		OaBool InHintNeedsPresentation = false
	);

	void Destroy();

	void PrintInfo() const;
	void PrintInfoCompact() const;
	void PrintInfoDetailed() const;
	void LogShaderPrecisionCaps() const;

	[[nodiscard]] OaBool NativeShaderBfloat16Usable() const noexcept {
		return Info.Software.ShaderBfloat16ExtensionEnabled && Info.Software.ShaderBfloat16TypeEnabled;
	}

	// Mesh: OaDevice::Index is the node index in OaDeviceMesh (0 = primary).
	void SetMeshLogicalIndex(OaI32 InMeshIndex) noexcept;
};


// ─────────────────────────────────────────────────────────────────────────────
// OaVkComputeDevice — Device with compute-specific capabilities
//
// Hierarchy: OaVkDevice → OaVkComputeDevice
// Adds: ML (CoopMat, BF16, IntDot, DGC), Vision (Video), Audio (future)
// ─────────────────────────────────────────────────────────────────────────────
class OaVkComputeDevice : public OaVkDevice {
public:
	// ─── ML Capabilities ───
	OaBool HasCooperativeMatrix          = false;
	OaBool HasCooperativeVector          = false;  // Blackwell+
	OaBool HasCooperativeMatrix2         = false;
	OaBool HasCooperativeMatrixDecodeVector = false;
	OaBool HasBFloat16                   = false;
	OaBool HasIntegerDotProduct          = false;
	OaBool HasDeviceGeneratedCommands    = false;
	OaVkCoopMatShapes CoopMatShapes;

	// ─── Vision Capabilities (compute-based) ───
	OaBool HasVideoDecodeQueue           = false;
	OaBool HasVideoEncodeQueue           = false;
	OaBool HasSamplerYcbcrConversion     = false;

	// ─── Audio Capabilities (compute-based) ───
	// Future: audio-specific compute features

	// ─── Vendor Trust Gates ───
	// Returns true if CoopMat is trustworthy on this vendor/device/driver combo.
	// Delegates to free function OaCoopMatTrust; bypassed by OA_FORCE_COOPMAT=1.
	[[nodiscard]] bool TrustCoopMatForVendor() const;

	// Returns true if BF16 is trustworthy on this vendor/device/driver combo.
	// Currently a pass-through to HasBFloat16; reserved for future blacklists.
	[[nodiscard]] bool TrustBf16ForVendor() const;

	// ─── Sync helpers ───
	// Copy class-level fields from Info.Software.* — call AFTER the trust gate
	// has run on Info.Software so the class fields reflect post-gate state.
	// This is the single source of truth for "is CoopMat actually usable here?"
	void SyncFromSoftwareInfo();

	// Log the discovered CoopMat shapes (gated by OA_LOG_COOPMAT_SHAPES env knob).
	// Called at device-init time by the engine's capability survey.
	void LogCoopMatShapes() const;

	// ─── Hardware Info ───
	// Returns SM/CU count for split-K GEMM decisions
	[[nodiscard]] OaU32 GetShaderCoreCount() const;
};


// ─────────────────────────────────────────────────────────────────────────────
// OaVkRenderDevice — Device with graphics/present capabilities
//
// Hierarchy: OaVkDevice → OaVkComputeDevice → OaVkRenderDevice
// Adds: Graphics queue, Present queue, Swapchain support
// ─────────────────────────────────────────────────────────────────────────────
class OaVkRenderDevice : public OaVkComputeDevice {
public:
	// ─── Graphics/Present Capabilities ───
	OaBool HasGraphicsQueue    = false;
	OaBool HasPresentQueue     = false;
	OaBool HasSwapchainSupport = false;

	// ─── Swapchain Helpers ───
	// Select optimal swapchain format for a surface
	[[nodiscard]] VkSurfaceFormatKHR SelectSwapchainFormat(VkSurfaceKHR InSurface) const;

	// Select optimal present mode for a surface
	[[nodiscard]] VkPresentModeKHR SelectPresentMode(VkSurfaceKHR InSurface) const;
};


// ─── Utility Functions ───────────────────────────────────────────────────────

OaDeviceType OaVkMapPhysicalType(VkPhysicalDeviceType InType);
const char*  OaVkPhysicalTypeLabel(VkPhysicalDeviceType InType);
const char*  OaVkVendorLabel(OaU32 InVendorId);
OaString     OaVkFormatDriverVersion(OaU32 InVersion);
const char*  OaVkDriverIdLabel(OaU32 InDriverId);
OaU32        OaVkCountComputeQueueSlots(void* InPhysicalDevice);

// OaCoopMatTrust is declared in <Oa/Runtime/Init.h> which Device.h includes.
// (Removed the redundant local re-declaration.)
