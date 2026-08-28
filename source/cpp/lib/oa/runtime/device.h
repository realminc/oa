// OA private vulkan logical device, queues, and capability probe.
// Physical enumeration pick score, estimates, and survey log: <oa/runtime/rate.h>.
// oavk::DeviceInfo groups oavk::DeviceHardwareInfo (PCI, VRAM, limits, heuristics) and
// oavk::DeviceSoftwareInfo (driver, API, enabled extensions, shader / extension caps).
#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/core/device.h>
#include <vkl/vkl.h>
#include <oa/runtime/init.h>     // oavk::CoopMatShape, oavk::CoopMatShapes
#include <oa/runtime/rate.h>

#include <oa/core/std/limits.h>

namespace oavk {

inline constexpr oa::U32 EnumerationIndexUnset = oa::Limits<oa::U32>::max();
inline constexpr oa::U32 PciDeviceIdUnknown    = oa::Limits<oa::U32>::max();


class Queues {
public:
	void* computeQueue            = nullptr;
	void* asyncComputeQueue       = nullptr;
	void* transferQueue           = nullptr;
	void* graphicsQueue           = nullptr;
	void* presentQueue            = nullptr;
	void* videoDecodeQueue        = nullptr;  // VK_KHR_video_decode_queue
	void* videoEncodeQueue        = nullptr;  // VK_KHR_video_encode_queue
	oa::U32 computeQueueFamily      = 0;
	oa::U32 asyncComputeQueueFamily = 0;
	oa::U32 transferQueueFamily     = 0;
	oa::U32 graphicsQueueFamily     = EnumerationIndexUnset;
	oa::U32 presentQueueFamily      = EnumerationIndexUnset;
	oa::U32 videoDecodeQueueFamily  = EnumerationIndexUnset;
	oa::U32 videoEncodeQueueFamily  = EnumerationIndexUnset;
	oa::Bool hasAsyncCompute        = false;
	oa::Bool hasPresentation        = false;
	oa::Bool hasVideoDecodeQueue    = false;
	oa::Bool hasVideoEncodeQueue    = false;
	// Per-family video codec operations (from VkQueueFamilyVideoPropertiesKHR).
	// Decoder create() checks these to verify the queue supports the target codec.
	VkVideoCodecOperationFlagsKHR videoDecodeCodecOps = 0;
	VkVideoCodecOperationFlagsKHR videoEncodeCodecOps = 0;
	oa::U32 computeQueueSlotCount           = 0;
	oa::U32 dedicatedTransferQueueSlotCount = 0;
};


class DeviceHardwareInfo {
public:
	oa::String     deviceName;
	oa::String     vendorName;
	oa::U32        vendorId                       = 0;
	oa::U32        deviceId                       = 0;
	oa::DeviceType deviceType                     = oa::DeviceType::Host;
	oa::U64        vramBytes                      = 0;
	oa::U32        subgroupSize                   = 0;
	oa::U32        maxComputeWorkGroupSize        = 0;
	oa::U32        maxComputeWorkGroupInvocations = 0;
	oa::U32        maxComputeWorkGroupCountX      = 0;
	oa::U32        maxComputeWorkGroupCountY      = 0;
	oa::U32        maxComputeWorkGroupCountZ      = 0;
	oa::U32        maxComputeSharedMemoryBytes    = 0;
	oa::U64        maxStorageBufferRangeBytes     = 0;
	oa::U32        numSMs                         = 0;  // streaming multiprocessors (NVIDIA) / cUs (AMD)
	oa::U32        maxPerStageDescriptorUpdateAfterBindStorageBuffers = 0;
	oa::U32        maxPerStageDescriptorUpdateAfterBindSampledImages = 0;
	oa::U32        maxPerStageDescriptorUpdateAfterBindSamplers = 0;
	// Engine-local descriptor capacities selected from the exact device limits.
	// These are requests; oavk::BindlessHeap records any smaller successful retry.
	oa::U32        bindlessBufferCapacity = 0;
	oa::U32        bindlessImageCapacity = 0;
	oa::U32        bindlessSamplerCapacity = 0;
	oa::Bool       hasSAM                         = false;
	oa::U64        pickRating                     = 0;
	oa::U32        enumerationIndex               = EnumerationIndexUnset;
	oa::F64        estMemBandwidthGbps            = 0.0;
	oa::F64        estPeakTflopsF32               = 0.0;
	oa::F64        timestampPeriodNanoseconds      = 0.0;
	oa::U32        computeTimestampValidBits       = 0;
};


class DeviceSoftwareInfo {
public:
	oa::String         driverVersion;
	oa::String         apiVersion;
	// Numeric physical-device API version. Runtime feature adapters may expose
	// newer behavior through extensions, but consumers such as VMA must still
	// receive the real core version advertised by the device.
	oa::U32            apiVersionPacked = VK_API_VERSION_1_0;
	oa::U32            driverId = 0;
	oa::String         driverName;
	oa::String         driverInfo;
	oa::Vector<oa::String>  enabledDeviceExtensions;
	
	oa::Bool hasCooperativeMatrix                    = false;
	oa::Bool hasCooperativeVector                    = false;  // vK_NV_cooperative_vector (Blackwell+)
	oa::Bool hasCooperativeMatrix2                   = false;  // VK_NV_cooperative_matrix2
	oa::Bool hasCooperativeMatrixDecodeVector        = false;  // VK_NV_cooperative_matrix_decode_vector
	// Discovered CoopMat shapes from vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR.
	// Populated when hasCooperativeMatrix is true. Future PR will thread these as slang
	// spec constants into fused-CoopMat kernels (replacing today's hardcoded 16x16x16).
	oavk::CoopMatShapes coopMatShapes;
	oa::Bool hasPipelineLibrary                      = false;
	oa::Bool has16BitStorage                         = false;
	oa::Bool shaderFloat16Enabled                    = false;
	oa::Bool shaderFloat64Enabled                    = false;
	oa::Bool shaderBfloat16ExtensionEnabled          = false;
	oa::Bool shaderBfloat16TypeEnabled               = false;
	oa::Bool shaderBfloat16DotProductEnabled         = false;
	oa::Bool shaderBfloat16CooperativeMatrixEnabled  = false;
	oa::Bool shaderIntegerDotProductEnabled          = false;  // vK_KHR_shader_integer_dot_product (INT8 quantization)
	oa::Bool hasExternalMemoryFd                     = false;
	oa::Bool hasKhrCalibratedTimestamps              = false;
	oa::Bool hasExtCalibratedTimestamps              = false;
	// exact presentation-resource retirement and binary-semaphore reuse proof.
	oa::Bool hasSwapchainMaintenance1                = false;
	
	// Device-generated commands (phase 2b DGC)
	oa::Bool hasDeviceGeneratedCommands              = false;  // vK_EXT_device_generated_commands (GPU-authored execution)
	
	// vulkan Video extensions (phase 2.1)
	oa::Bool hasVideoQueue                           = false;  // VK_KHR_video_queue
	oa::Bool hasVideoDecodeQueue                     = false;  // VK_KHR_video_decode_queue
	oa::Bool hasVideoDecodeH264                      = false;  // VK_KHR_video_decode_h264
	oa::Bool hasVideoDecodeH265                      = false;  // VK_KHR_video_decode_h265
	oa::Bool hasVideoDecodeAV1                       = false;  // VK_KHR_video_decode_av1
	oa::Bool hasVideoDecodeVP9                     = false;  // VK_KHR_video_decode_vp9
	oa::Bool hasVideoEncodeQueue                     = false;  // VK_KHR_video_encode_queue
	oa::Bool hasVideoEncodeH264                      = false;  // VK_KHR_video_encode_h264
	oa::Bool hasVideoEncodeH265                      = false;  // VK_KHR_video_encode_h265
	oa::Bool hasVideoEncodeAV1                       = false;  // VK_KHR_video_encode_av1
	oa::Bool hasSamplerYcbcrConversion               = false;  // VK_KHR_sampler_ycbcr_conversion
};


class DeviceInfo {
public:
	DeviceHardwareInfo hardware;
	DeviceSoftwareInfo software;
};


class Device : public oa::Device {
public:
	// vulkan implementation of a logical oa::Device: base holds type + mesh index.

	// Data
	void* instance       = nullptr;
	void* physicalDevice = nullptr;
	void* device         = nullptr;
	VklInstanceTable instanceDispatch{};
	VklDeviceTable deviceDispatch{};
	oa::Bool ownsInstance  = true;
	Queues     queues;
	DeviceInfo info;

	// -----------------------------------------------------------------------
	// Create — single physical device.
	//
	// inHintNeedsPresentation:
	//   Pass true when the caller intends to attach a VkSurfaceKHR later (e.g.
	//   via oa::Presenter::initPresentation).  The device will be created
	//   with VK_QUEUE_GRAPHICS_BIT + VK_KHR_swapchain without needing the
	//   surface at creation time, so SDL_Vulkan_CreateSurface can be called
	//   against the resulting instance handle.
	// inHintNeedsGraphics:
	//   Pass true for headless graphics. A graphics-capable queue is requested,
	//   but no surface or swapchain extension is enabled.
	// -----------------------------------------------------------------------
	[[nodiscard]] static oa::Result<Device> create(
		oa::StringView              inAppName,
		oa::Bool                    inEnableValidation,
		oa::DeviceType              inPreferred               = oa::DeviceType::VkDiscrete,
		oa::U32                     inForceEnumerationIndex   = EnumerationIndexUnset,
		oa::U32                     inAppVersionPatch         = 1,
		oa::Span<const char* const> inInstanceExtraExtensions = {},
		oa::Bool                    inHintNeedsPresentation   = false,
		oa::Bool                    inHintNeedsGraphics       = false,
		PFN_vkGetInstanceProcAddr   inCustomLoader            = nullptr
	);

	// -----------------------------------------------------------------------
	// CreateFromPhysical — re-use an existing VkInstance.
	//
	// inSurface (VkSurfaceKHR):
	//   Legacy compatibility parameter. The base-device path does not inspect
	//   the surface; use oavk::DeviceBuilder::BuildRender for surface-aware queue
	//   planning. inHintNeedsPresentation still prepares this base device for
	//   the later oa::Presenter attachment path.
	//
	// inHintNeedsPresentation:
	//   When true and inSurface is null, a graphics-capable queue family and
	//   VK_KHR_swapchain are still enabled so the device is ready for
	//   swapchain creation once a surface becomes available.
	//
	// inHintNeedsGraphics:
	//   When true and no presentation is requested, selects a graphics-capable
	//   queue without enabling any surface-dependent extension.
	// -----------------------------------------------------------------------
	[[nodiscard]] static oa::Result<Device> createFromPhysical(
		void*  inInstance,
		void*  inPhysicalDevice,
		oa::Bool inEnableValidation,
		oa::U64  inPickRating            = 0,
		oa::U32  inEnumerationIndex      = EnumerationIndexUnset,
		void*  inSurface               = nullptr,
		oa::Bool inHintNeedsPresentation = false,
		oa::Bool inHintNeedsGraphics     = false
	);

	void destroy();

	void printInfo() const;
	void printInfoCompact() const;
	void printInfoDetailed() const;
	void logShaderPrecisionCaps() const;

	[[nodiscard]] oa::Bool nativeShaderBfloat16Usable() const noexcept {
		return info.software.shaderBfloat16ExtensionEnabled && info.software.shaderBfloat16TypeEnabled;
	}

};


// ─────────────────────────────────────────────────────────────────────────────
// oavk::ComputeDevice — Device with compute-specific capabilities
//
// Hierarchy: oavk::Device → oavk::ComputeDevice
// Adds: ML (CoopMat, BF16, IntDot, DGC), vision (Video), Audio (future)
// ─────────────────────────────────────────────────────────────────────────────
class ComputeDevice : public Device {
public:
	// ─── ML capabilities ───
	oa::Bool hasCooperativeMatrix          = false;
	oa::Bool hasCooperativeVector          = false;  // Blackwell+
	oa::Bool hasCooperativeMatrix2         = false;
	oa::Bool hasCooperativeMatrixDecodeVector = false;
	oa::Bool HasBFloat16                   = false;
	oa::Bool HasIntegerDotProduct          = false;
	oa::Bool hasDeviceGeneratedCommands    = false;
	oavk::CoopMatShapes coopMatShapes;

	// ─── Vision capabilities (compute-based) ───
	oa::Bool hasVideoDecodeQueue           = false;
	oa::Bool hasVideoEncodeQueue           = false;
	oa::Bool hasSamplerYcbcrConversion     = false;

	// ─── Audio capabilities (compute-based) ───
	// Future: audio-specific compute features

	// ─── Vendor Trust Gates ───
	// Returns true if CoopMat is trustworthy on this vendor/device/driver combo.
	// Delegates to free function oavk::coopMatTrust; bypassed by OA_FORCE_COOPMAT=1.
	[[nodiscard]] bool trustCoopMatForVendor() const;

	// Returns true if BF16 is trustworthy on this vendor/device/driver combo.
	// Currently a pass-through to HasBFloat16; reserved for future blacklists.
	[[nodiscard]] bool trustBf16ForVendor() const;

	// ─── Sync helpers ───
	// Copy class-level fields from Info.software.* — call AFTER the trust gate
	// has run on Info.software so the class fields reflect post-gate state.
	// This is the single source of truth for "is CoopMat actually usable here?"
	void syncFromSoftwareInfo();

	// log the discovered CoopMat shapes (gated by OA_LOG_COOPMAT_SHAPES env knob).
	// Called at device-init time by the engine's capability survey.
	void logCoopMatShapes() const;

	// ─── hardware Info ───
	// Returns SM/CU count for split-K GEMM decisions
	[[nodiscard]] oa::U32 getShaderCoreCount() const;
};


// ─────────────────────────────────────────────────────────────────────────────
// oavk::RenderDevice — Device with graphics/present capabilities
//
// Hierarchy: oavk::Device → oavk::ComputeDevice → oavk::RenderDevice
// Adds: Graphics queue, Present queue, Swapchain support
// ─────────────────────────────────────────────────────────────────────────────
class RenderDevice : public ComputeDevice {
public:
	// ─── Graphics/Present capabilities ───
	oa::Bool hasGraphicsQueue    = false;
	oa::Bool hasPresentQueue     = false;
	oa::Bool hasSwapchainSupport = false;

	// ─── Swapchain Helpers ───
	// Select optimal swapchain format for a surface
	[[nodiscard]] VkSurfaceFormatKHR selectSwapchainFormat(VkSurfaceKHR inSurface) const;

	// Select optimal present mode for a surface
	[[nodiscard]] VkPresentModeKHR selectPresentMode(VkSurfaceKHR inSurface) const;
};


// ─── Utility Functions ───────────────────────────────────────────────────────

oa::DeviceType mapPhysicalType(VkPhysicalDeviceType inType);
const char* physicalTypeLabel(VkPhysicalDeviceType inType);
const char* vendorLabel(oa::U32 inVendorId);
oa::String formatDriverVersion(oa::U32 inVersion);
const char* driverIdLabel(oa::U32 inDriverId);
oa::U32 countComputeQueueSlots(const VklInstanceTable& inDispatch, void* inPhysicalDevice);

} // namespace oavk

// oavk::coopMatTrust is declared in <oa/runtime/init.h>, included by device.h.
// (Removed the redundant local re-declaration.)
