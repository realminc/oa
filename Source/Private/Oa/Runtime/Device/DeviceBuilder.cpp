// OA Vulkan Device Builder Implementation
#include "DeviceBuilder.h"
#include <Oa/Core/Log.h>
#include <Oa/Core/EnvFlag.h>
#include <Oa/Runtime/Init.h>
#include <Oa/Runtime/Bindless.h>
#include <algorithm>

// Forward declarations from Device.cpp
OaStatus OaVkPlanDeviceQueues(
	VkPhysicalDevice InPhys,
	VkSurfaceKHR InSurface,
	OaVkQueuePlan& OutPlan,
	bool InNeedsGraphics
);


// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Module Registration
// ─────────────────────────────────────────────────────────────────────────────

OaVkDeviceBuilder& OaVkDeviceBuilder::WithCore() {
	if (!HasCoreModule_) {
		Modules_.PushBack(OaVkCreateCoreFeatures());
		HasCoreModule_ = true;
	}
	return *this;
}


OaVkDeviceBuilder& OaVkDeviceBuilder::WithMl() {
	if (!HasMlModule_) {
		Modules_.PushBack(OaVkCreateMlFeatures());
		HasMlModule_ = true;
	}
	return *this;
}


OaVkDeviceBuilder& OaVkDeviceBuilder::WithVision() {
	if (!HasVisionModule_) {
		Modules_.PushBack(OaVkCreateVisionFeatures());
		HasVisionModule_ = true;
	}
	return *this;
}


OaVkDeviceBuilder& OaVkDeviceBuilder::WithAudio() {
	if (!HasAudioModule_) {
		Modules_.PushBack(OaVkCreateAudioFeatures());
		HasAudioModule_ = true;
	}
	return *this;
}


OaVkDeviceBuilder& OaVkDeviceBuilder::WithRender() {
	if (!HasRenderModule_) {
		Modules_.PushBack(OaVkCreateRenderFeatures());
		HasRenderModule_ = true;
	}
	return *this;
}


OaVkDeviceBuilder& OaVkDeviceBuilder::WithAllCompute() {
	return WithCore().WithMl().WithVision().WithAudio();
}


OaVkDeviceBuilder& OaVkDeviceBuilder::WithAllFeatures() {
	return WithCore().WithMl().WithVision().WithAudio().WithRender();
}


// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Probe Extensions
// ─────────────────────────────────────────────────────────────────────────────

void OaVkDeviceBuilder::ProbeExtensions(const OaVec<VkExtensionProperties>& InExtensions) {
	for (const auto& module : Modules_) {
		module->ProbeExtensions(InExtensions, ExtProbe_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Query Features
// ─────────────────────────────────────────────────────────────────────────────

void OaVkDeviceBuilder::QueryFeatures(VkPhysicalDevice InPhysicalDevice) {
	for (const auto& module : Modules_) {
		module->QueryFeatures(InPhysicalDevice, FeatureBundle_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Build Feature Chains
// ─────────────────────────────────────────────────────────────────────────────

void OaVkDeviceBuilder::BuildFeatureChains() {
	for (const auto& module : Modules_) {
		module->BuildFeatureChain(FeatureBundle_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// Phase 5: Collect Extensions
// ─────────────────────────────────────────────────────────────────────────────

void OaVkDeviceBuilder::CollectExtensions() {
	EnabledExtensions_.Clear();
	for (const auto& module : Modules_) {
		module->CollectExtensions(ExtProbe_, FeatureBundle_, EnabledExtensions_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// Dependency Validation
// ─────────────────────────────────────────────────────────────────────────────

OaStatus OaVkDeviceBuilder::ValidateDependencies() const {
	// Build module name map
	OaHashSet<OaStringView> availableModules;
	for (const auto& module : Modules_) {
		availableModules.Insert(module->Name());
	}

	// Check each module's dependencies
	for (const auto& module : Modules_) {
		const auto deps = module->Dependencies();
		for (const auto& dep : deps) {
			if (!availableModules.Contains(dep)) {
				return OaStatus::Error(
					OaStatusCode::FailedPrecondition,
					OaString("Module '") + OaString(module->Name()) +
					"' requires module '" + OaString(dep) + "' which is not loaded"
				);
			}
		}
	}

	return OaStatus::Ok();
}


void OaVkDeviceBuilder::SortModulesByDependencies() {
	// Simple topological sort: move modules with no dependencies first
	// This is sufficient for our small module count
	std::stable_sort(Modules_.Begin(), Modules_.End(),
		[](const OaUniquePtr<OaVkFeatureModule>& a, const OaUniquePtr<OaVkFeatureModule>& b) {
			return a->Dependencies().Size() < b->Dependencies().Size();
		}
	);
}


// ─────────────────────────────────────────────────────────────────────────────
// Phase 8: Type-Safe Device Creation
// ─────────────────────────────────────────────────────────────────────────────

OaResult<OaVkDevice> OaVkDeviceBuilder::BuildBase(
	VkInstance InInstance,
	VkPhysicalDevice InPhysicalDevice,
	OaBool InEnableValidation,
	OaBool InWantsPresentation,
	OaBool InNeedsGraphics
) {
	// Validate Core module is present
	if (!HasCoreModule_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition,
			"BuildBase requires WithCore()");
	}

	// Validate and sort modules
	OaStatus depStatus = ValidateDependencies();
	if (!depStatus.IsOk()) {
		return depStatus;
	}
	SortModulesByDependencies();

	// Enumerate extensions
	OaU32 extCount = 0;
	vkEnumerateDeviceExtensionProperties(InPhysicalDevice, nullptr, &extCount, nullptr);
	OaVec<VkExtensionProperties> extensions(extCount);
	vkEnumerateDeviceExtensionProperties(InPhysicalDevice, nullptr, &extCount, extensions.Data());

	// Run build pipeline
	ProbeExtensions(extensions);
	QueryFeatures(InPhysicalDevice);
	if (!InWantsPresentation) {
		FeatureBundle_.HasSwapchainMaintenance1 = false;
	}
	BuildFeatureChains();
	CollectExtensions();
	// WithAllFeatures() is also used by the headless compute context.  Merely
	// loading the render feature module must not enable VK_KHR_swapchain: the
	// extension requires the instance-side surface extension even when no
	// swapchain is ever created (VUID-vkCreateDevice-ppEnabledExtensionNames-01387).
	if (!InWantsPresentation) {
		for (OaUsize i = 0; i < EnabledExtensions_.Size();) {
			if (strcmp(EnabledExtensions_[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
				EnabledExtensions_.Erase(EnabledExtensions_.Begin() + i);
			} else {
				++i;
			}
		}
	}

	// Plan queues. When the caller hinted at later swapchain attachment,
	// the planner picks a graphics-capable family even though InSurface is
	// null here (verified against the real surface during InitPresentation).
	OaVkQueuePlan queuePlan;
	OaStatus queueStatus = OaVkPlanDeviceQueues(
		InPhysicalDevice, VK_NULL_HANDLE, queuePlan,
		InWantsPresentation or InNeedsGraphics);
	if (!queueStatus.IsOk()) {
		return queueStatus;
	}

	// Create logical device
	auto deviceResult = CreateLogicalDevice(InPhysicalDevice, queuePlan);
	if (!deviceResult.IsOk()) {
		return deviceResult.GetStatus();
	}

	// Populate device structure
	OaVkDevice device;
	device.Instance = InInstance;
	device.PhysicalDevice = InPhysicalDevice;
	device.Device = deviceResult.GetValue();
	device.OwnsInstance = false;

	// Retrieve queue handles
	VkDevice vkDevice = deviceResult.GetValue();
	VkQueue computeQ = VK_NULL_HANDLE;
	VkQueue asyncComputeQ = VK_NULL_HANDLE;
	VkQueue transferQ = VK_NULL_HANDLE;

	vkGetDeviceQueue(vkDevice, queuePlan.ComputeQF, 0, &computeQ);
	vkGetDeviceQueue(vkDevice, queuePlan.TransferQF, 0, &transferQ);

	OaU32 actualAsyncQF = queuePlan.ComputeQF;
	if (queuePlan.AsyncComputeQF != UINT32_MAX) {
		vkGetDeviceQueue(vkDevice, queuePlan.AsyncComputeQF, 0, &asyncComputeQ);
		actualAsyncQF = queuePlan.AsyncComputeQF;
	} else if (queuePlan.ComputeHasMultiQueue) {
		vkGetDeviceQueue(vkDevice, queuePlan.ComputeQF, 1, &asyncComputeQ);
		actualAsyncQF = queuePlan.ComputeQF;
	}

	// Populate queue info
	device.Queues.ComputeQueue = computeQ;
	device.Queues.AsyncComputeQueue = asyncComputeQ;
	device.Queues.TransferQueue = transferQ;
	device.Queues.ComputeQueueFamily = queuePlan.ComputeQF;
	device.Queues.AsyncComputeQueueFamily = actualAsyncQF;
	device.Queues.TransferQueueFamily = queuePlan.TransferQF;
	device.Queues.HasAsyncCompute = queuePlan.HasAsync;
	device.Queues.ComputeQueueSlotCount = queuePlan.ComputeSlots;
	device.Queues.DedicatedTransferQueueSlotCount = queuePlan.DedicatedTransferSlots;

	// Step 3g.1.b followup: when the planner found video-decode / video-encode
	// queue families on the device, fetch their VkQueue handles now and
	// publish them on device.Queues so OaVideoDecoder / OaVideoEncoder can
	// submit on the right queue. Without this, the planner would bump the
	// queueCount for these families in the VkDeviceCreateInfo but the queue
	// handles were never extracted — IsCodecSupported returned false even on
	// devices that genuinely exposed the extensions.
	if (queuePlan.VideoDecodeQF != UINT32_MAX) {
		VkQueue videoDecodeQ = VK_NULL_HANDLE;
		vkGetDeviceQueue(vkDevice, queuePlan.VideoDecodeQF, 0, &videoDecodeQ);
		device.Queues.VideoDecodeQueue        = videoDecodeQ;
		device.Queues.VideoDecodeQueueFamily  = queuePlan.VideoDecodeQF;
		device.Queues.HasVideoDecodeQueue     = (videoDecodeQ != VK_NULL_HANDLE);
		device.Queues.VideoDecodeCodecOps     = queuePlan.VideoDecodeCodecOps;
	}
	if (queuePlan.VideoEncodeQF != UINT32_MAX) {
		VkQueue videoEncodeQ = VK_NULL_HANDLE;
		vkGetDeviceQueue(vkDevice, queuePlan.VideoEncodeQF, 0, &videoEncodeQ);
		device.Queues.VideoEncodeQueue        = videoEncodeQ;
		device.Queues.VideoEncodeQueueFamily  = queuePlan.VideoEncodeQF;
		device.Queues.HasVideoEncodeQueue     = (videoEncodeQ != VK_NULL_HANDLE);
		device.Queues.VideoEncodeCodecOps     = queuePlan.VideoEncodeCodecOps;
	}

	// A graphics queue is useful independently of WSI. Presentation queue state
	// remains unset until OaPresenter verifies the actual surface.
	if (queuePlan.WantsGraphics
		and queuePlan.GraphicsQF != UINT32_MAX) {
		VkQueue graphicsQ = VK_NULL_HANDLE;
		vkGetDeviceQueue(vkDevice, queuePlan.GraphicsQF, 0, &graphicsQ);
		device.Queues.GraphicsQueue        = graphicsQ;
		device.Queues.GraphicsQueueFamily  = queuePlan.GraphicsQF;
	}

	PopulateDeviceInfo(InPhysicalDevice, deviceResult.GetValue(), device);

	return device;
}


OaResult<OaVkComputeDevice> OaVkDeviceBuilder::BuildCompute(
	VkInstance InInstance,
	VkPhysicalDevice InPhysicalDevice,
	OaBool InEnableValidation
) {
	// Build base device first
	auto baseResult = BuildBase(InInstance, InPhysicalDevice, InEnableValidation);
	if (!baseResult.IsOk()) {
		return baseResult.GetStatus();
	}

	// Upcast to compute device
	OaVkComputeDevice computeDevice;
	static_cast<OaVkDevice&>(computeDevice) = baseResult.GetValue();

	// Populate compute-class fields from Info.Software (already trust-gated by
	// BuildBase). Previously this block read directly from FeatureBundle_,
	// which bypassed the vendor trust gate — e.g. on an untrusted device,
	// Info.Software.HasCooperativeMatrix would be false but
	// computeDevice.HasCooperativeMatrix would silently stay true.
	computeDevice.SyncFromSoftwareInfo();
	computeDevice.LogCoopMatShapes();

	return computeDevice;
}


OaResult<OaVkRenderDevice> OaVkDeviceBuilder::BuildRender(
	VkInstance InInstance,
	VkPhysicalDevice InPhysicalDevice,
	OaBool InEnableValidation,
	VkSurfaceKHR InSurface
) {
	// Validate Render module is present
	if (!HasRenderModule_) {
		return OaStatus::Error(OaStatusCode::FailedPrecondition, "BuildRender requires WithRender()");
	}

	// Validate and sort modules
	OaStatus depStatus = ValidateDependencies();
	if (!depStatus.IsOk()) {
		return depStatus;
	}
	SortModulesByDependencies();

	// Enumerate extensions
	OaU32 extCount = 0;
	vkEnumerateDeviceExtensionProperties(InPhysicalDevice, nullptr, &extCount, nullptr);
	OaVec<VkExtensionProperties> extensions(extCount);
	vkEnumerateDeviceExtensionProperties(InPhysicalDevice, nullptr, &extCount, extensions.Data());

	// Run build pipeline
	ProbeExtensions(extensions);
	QueryFeatures(InPhysicalDevice);
	if (InSurface == VK_NULL_HANDLE) {
		FeatureBundle_.HasSwapchainMaintenance1 = false;
	}
	BuildFeatureChains();
	CollectExtensions();
	if (InSurface == VK_NULL_HANDLE) {
		for (OaUsize i = 0; i < EnabledExtensions_.Size();) {
			if (strcmp(EnabledExtensions_[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
				EnabledExtensions_.Erase(EnabledExtensions_.Begin() + i);
			} else {
				++i;
			}
		}
	}

	// Plan queues with graphics/present support
	OaVkQueuePlan queuePlan;
	bool wantsPresentation = (InSurface != VK_NULL_HANDLE);
	OaStatus queueStatus = OaVkPlanDeviceQueues(
		InPhysicalDevice, InSurface, queuePlan, wantsPresentation);
	if (!queueStatus.IsOk()) {
		return queueStatus;
	}

	// Create logical device
	auto deviceResult = CreateLogicalDevice(InPhysicalDevice, queuePlan);
	if (!deviceResult.IsOk()) {
		return deviceResult.GetStatus();
	}

	// Build base device structure
	OaVkRenderDevice renderDevice;
	renderDevice.Instance = InInstance;
	renderDevice.PhysicalDevice = InPhysicalDevice;
	renderDevice.Device = deviceResult.GetValue();
	renderDevice.OwnsInstance = false;

	// Retrieve queue handles (compute + graphics/present)
	VkDevice vkDevice = deviceResult.GetValue();
	VkQueue computeQ = VK_NULL_HANDLE;
	VkQueue asyncComputeQ = VK_NULL_HANDLE;
	VkQueue transferQ = VK_NULL_HANDLE;
	VkQueue graphicsQ = VK_NULL_HANDLE;
	VkQueue presentQ = VK_NULL_HANDLE;
	
	vkGetDeviceQueue(vkDevice, queuePlan.ComputeQF, 0, &computeQ);
	vkGetDeviceQueue(vkDevice, queuePlan.TransferQF, 0, &transferQ);
	
	OaU32 actualAsyncQF = queuePlan.ComputeQF;
	if (queuePlan.AsyncComputeQF != UINT32_MAX) {
		vkGetDeviceQueue(vkDevice, queuePlan.AsyncComputeQF, 0, &asyncComputeQ);
		actualAsyncQF = queuePlan.AsyncComputeQF;
	} else if (queuePlan.ComputeHasMultiQueue) {
		vkGetDeviceQueue(vkDevice, queuePlan.ComputeQF, 1, &asyncComputeQ);
		actualAsyncQF = queuePlan.ComputeQF;
	}
	
	if (queuePlan.WantsGraphics) {
		vkGetDeviceQueue(vkDevice, queuePlan.GraphicsQF, 0, &graphicsQ);
		if (InSurface != VK_NULL_HANDLE) presentQ = graphicsQ;
	}
	
	// Populate queue info
	renderDevice.Queues.ComputeQueue = computeQ;
	renderDevice.Queues.AsyncComputeQueue = asyncComputeQ;
	renderDevice.Queues.TransferQueue = transferQ;
	renderDevice.Queues.GraphicsQueue = graphicsQ;
	renderDevice.Queues.PresentQueue = presentQ;
	renderDevice.Queues.ComputeQueueFamily = queuePlan.ComputeQF;
	renderDevice.Queues.AsyncComputeQueueFamily = actualAsyncQF;
	renderDevice.Queues.TransferQueueFamily = queuePlan.TransferQF;
	renderDevice.Queues.GraphicsQueueFamily = queuePlan.GraphicsQF;
	renderDevice.Queues.PresentQueueFamily = queuePlan.PresentQF;
	renderDevice.Queues.HasAsyncCompute = queuePlan.HasAsync;
	renderDevice.Queues.HasPresentation = (InSurface != VK_NULL_HANDLE) && (graphicsQ != VK_NULL_HANDLE);
	renderDevice.Queues.ComputeQueueSlotCount = queuePlan.ComputeSlots;
	renderDevice.Queues.DedicatedTransferQueueSlotCount = queuePlan.DedicatedTransferSlots;

	// Same video-queue fix as BuildBase — fetch the video-decode/encode
	// VkQueue handles when the planner found their families.
	if (queuePlan.VideoDecodeQF != UINT32_MAX) {
		VkQueue videoDecodeQ = VK_NULL_HANDLE;
		vkGetDeviceQueue(vkDevice, queuePlan.VideoDecodeQF, 0, &videoDecodeQ);
		renderDevice.Queues.VideoDecodeQueue        = videoDecodeQ;
		renderDevice.Queues.VideoDecodeQueueFamily  = queuePlan.VideoDecodeQF;
		renderDevice.Queues.HasVideoDecodeQueue     = (videoDecodeQ != VK_NULL_HANDLE);
		renderDevice.Queues.VideoDecodeCodecOps     = queuePlan.VideoDecodeCodecOps;
	}
	if (queuePlan.VideoEncodeQF != UINT32_MAX) {
		VkQueue videoEncodeQ = VK_NULL_HANDLE;
		vkGetDeviceQueue(vkDevice, queuePlan.VideoEncodeQF, 0, &videoEncodeQ);
		renderDevice.Queues.VideoEncodeQueue        = videoEncodeQ;
		renderDevice.Queues.VideoEncodeQueueFamily  = queuePlan.VideoEncodeQF;
		renderDevice.Queues.HasVideoEncodeQueue     = (videoEncodeQ != VK_NULL_HANDLE);
		renderDevice.Queues.VideoEncodeCodecOps     = queuePlan.VideoEncodeCodecOps;
	}

	// Populate device info (also applies the CoopMat vendor trust gate to
	// renderDevice.Info.Software — see PopulateDeviceInfo body).
	PopulateDeviceInfo(InPhysicalDevice, vkDevice, static_cast<OaVkDevice&>(renderDevice));

	// Mirror trust-gated Info.Software.* into class-level fields. Same fix as
	// BuildCompute — previously read from FeatureBundle_ which bypasses the
	// gate.
	renderDevice.SyncFromSoftwareInfo();

	// Populate render-specific capabilities
	renderDevice.HasGraphicsQueue    = graphicsQ != VK_NULL_HANDLE;
	renderDevice.HasPresentQueue     = presentQ  != VK_NULL_HANDLE;
	renderDevice.HasSwapchainSupport = ExtProbe_.KhrSwapchain;
	renderDevice.Info.Software.HasSwapchainMaintenance1 =
		FeatureBundle_.HasSwapchainMaintenance1 && InSurface != VK_NULL_HANDLE;

	return renderDevice;
}


// ─────────────────────────────────────────────────────────────────────────────
// Helper: Create Logical Device
// ─────────────────────────────────────────────────────────────────────────────

OaResult<VkDevice> OaVkDeviceBuilder::CreateLogicalDevice(
	VkPhysicalDevice InPhysicalDevice,
	const OaVkQueuePlan& InQueuePlan
) {
	VkDeviceCreateInfo devCI{};
	devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	devCI.pNext = &FeatureBundle_.Features2;
	devCI.queueCreateInfoCount = static_cast<OaU32>(InQueuePlan.QueueCIs.Size());
	devCI.pQueueCreateInfos = InQueuePlan.QueueCIs.Data();
	devCI.enabledExtensionCount = static_cast<OaU32>(EnabledExtensions_.Size());
	devCI.ppEnabledExtensionNames = EnabledExtensions_.Data();

	VkDevice device = VK_NULL_HANDLE;
	VkResult r = vkCreateDevice(InPhysicalDevice, &devCI, nullptr, &device);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,
			"vkCreateDevice failed");
	}

	return device;
}


// ─────────────────────────────────────────────────────────────────────────────
// Helper: Populate Device Info
// ─────────────────────────────────────────────────────────────────────────────

void OaVkDeviceBuilder::PopulateDeviceInfo(
	VkPhysicalDevice InPhysicalDevice,
	VkDevice InDevice,
	OaVkDevice& OutDevice
) {
	// Get physical device properties
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(InPhysicalDevice, &props);

	// Populate hardware info
	OutDevice.Info.Hardware.DeviceName = props.deviceName;
	OutDevice.Info.Hardware.VendorId = props.vendorID;
	OutDevice.Info.Hardware.DeviceId = props.deviceID;
	OutDevice.Info.Hardware.DeviceType = OaVkMapPhysicalType(props.deviceType);
	OutDevice.Info.Hardware.VendorName = OaVkVendorLabel(props.vendorID);

	// Get memory info
	VkPhysicalDeviceMemoryProperties memProps{};
	vkGetPhysicalDeviceMemoryProperties(InPhysicalDevice, &memProps);
	OaU64 vram = 0;
	for (OaU32 i = 0; i < memProps.memoryHeapCount; i++) {
		if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			vram += memProps.memoryHeaps[i].size;
		}
	}
	OutDevice.Info.Hardware.VramBytes = vram;

	// Get subgroup size and driver info
	VkPhysicalDeviceSubgroupProperties subgroupProps{};
	subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	VkPhysicalDeviceDriverProperties driverProps{};
	driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
	subgroupProps.pNext = &driverProps;
	VkPhysicalDeviceProperties2 props2{};
	props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2.pNext = &subgroupProps;
	vkGetPhysicalDeviceProperties2(InPhysicalDevice, &props2);
	
	OutDevice.Info.Hardware.SubgroupSize = subgroupProps.subgroupSize;
	OutDevice.Info.Hardware.MaxComputeWorkGroupSize = props.limits.maxComputeWorkGroupSize[0];
	OutDevice.Info.Hardware.MaxComputeWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
	OutDevice.Info.Hardware.MaxComputeSharedMemoryBytes = props.limits.maxComputeSharedMemorySize;

	// Query descriptor indexing limits for bindless heap capacity
	VkPhysicalDeviceDescriptorIndexingProperties indexingProps{};
	indexingProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
	VkPhysicalDeviceProperties2 props2ForIndexing{};
	props2ForIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2ForIndexing.pNext = &indexingProps;
	vkGetPhysicalDeviceProperties2(InPhysicalDevice, &props2ForIndexing);

	// Store the full reported limits for logging
	OutDevice.Info.Hardware.MaxPerStageDescriptorUpdateAfterBindStorageBuffers =
		static_cast<OaU32>(indexingProps.maxPerStageDescriptorUpdateAfterBindStorageBuffers);
	OutDevice.Info.Hardware.MaxPerStageDescriptorUpdateAfterBindSampledImages =
		static_cast<OaU32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages);
	OutDevice.Info.Hardware.MaxPerStageDescriptorUpdateAfterBindSamplers =
		static_cast<OaU32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSamplers);

	// Set bindless capacity based on device limits with safe caps.
	//
	// Buffers: iGPUs report 100M+ theoretical UAB storage-buffer slots but cannot
	// back them efficiently. The descriptor pool AND our slot free-list are sized to
	// capacity, so an unclamped request means multi-second/minute init plus OOM on a
	// shared-memory ultrabook. Discrete GPUs are naturally bounded near 1M, so we cap
	// there; integrated / CPU devices get a tighter 256K cap (still vastly more live
	// tensors than any single ML graph needs on an iGPU). Either can be overridden at
	// runtime with OA_BINDLESS_BUFFER_CAP=N for debugging or large-model experiments.
	// Images capped to 16K (pure compute) and samplers to 2K to avoid pipeline stalls.
	constexpr OaU32 kSafeBufferLimitDiscrete   = 1048576;   // 1M
	constexpr OaU32 kSafeBufferLimitIntegrated = 262144;    // 256K
	constexpr OaU32 kSafeImageLimit            = 16384;     // 16K
	constexpr OaU32 kSafeSamplerLimit          = 2048;      // 2K

	const bool isIntegrated =
		OutDevice.Info.Hardware.DeviceType == OaDeviceType::VkIntegrated ||
		OutDevice.Info.Hardware.DeviceType == OaDeviceType::VkCpu;
	OaU32 bufferLimit = isIntegrated ? kSafeBufferLimitIntegrated : kSafeBufferLimitDiscrete;

	// Runtime override: OA_BINDLESS_BUFFER_CAP=N (later clamped to the reported limit).
	const OaI64 envBufferCap = OaEnvFlag::GetInt("OA_BINDLESS_BUFFER_CAP", 0);
	if (envBufferCap > 0) {
		bufferLimit = static_cast<OaU32>(envBufferCap);
		OA_LOG_INFO(OaLogComponent::Core,
			"Bindless: buffer cap overridden via OA_BINDLESS_BUFFER_CAP=%lld",
			static_cast<long long>(envBufferCap));
	}

	OA_BINDLESS_CAPACITY = std::min(
		static_cast<OaU32>(indexingProps.maxPerStageDescriptorUpdateAfterBindStorageBuffers),
		bufferLimit
	);
	OA_BINDLESS_IMAGE_CAPACITY = std::min(
		static_cast<OaU32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages),
		kSafeImageLimit
	);
	OA_BINDLESS_SAMPLER_CAPACITY = std::min(
		static_cast<OaU32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSamplers),
		kSafeSamplerLimit
	);

	// Ensure minimum capacities (fallback for devices that report 0 or very low values)
	OA_BINDLESS_CAPACITY = std::max(OA_BINDLESS_CAPACITY, OA_BINDLESS_CAPACITY_FALLBACK);
	OA_BINDLESS_IMAGE_CAPACITY = std::max(OA_BINDLESS_IMAGE_CAPACITY, OA_BINDLESS_IMAGE_CAPACITY_FALLBACK);
	OA_BINDLESS_SAMPLER_CAPACITY = std::max(OA_BINDLESS_SAMPLER_CAPACITY, OA_BINDLESS_SAMPLER_CAPACITY_FALLBACK);
	
	// Detect SAM (Smart Access Memory / Resizable BAR)
	bool hasSam = false;
	for (OaU32 i = 0; i < memProps.memoryTypeCount; i++) {
		if ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
			(memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
		{
			OaU32 heapIdx = memProps.memoryTypes[i].heapIndex;
			if (memProps.memoryHeaps[heapIdx].size > 256ULL * 1024 * 1024) {
				hasSam = true;
				break;
			}
		}
	}
	OutDevice.Info.Hardware.HasSAM = hasSam;
	
	// Estimate SM/CU count
	OutDevice.Info.Hardware.NumSMs = OaVkEstimateNumSMsEx(
		props.vendorID, props.deviceID, OutDevice.Info.Hardware.DeviceType, vram);
	
	// Estimate performance metrics
	OutDevice.Info.Hardware.EstMemBandwidthGbps = OaVkEstimateMemBandwidthGbpsEx(
		props.vendorID, props.deviceID, OutDevice.Info.Hardware.DeviceType, vram);
	OutDevice.Info.Hardware.EstPeakTflopsF32 = OaVkEstimatePeakTflopsF32Ex(
		props.vendorID, props.deviceID, OutDevice.Info.Hardware.DeviceType, vram);
	
	// Populate software info
	OutDevice.Info.Software.DriverVersion = OaVkFormatDriverVersion(props.driverVersion);
	OutDevice.Info.Software.ApiVersion = OaVkFormatDriverVersion(props.apiVersion);
	OutDevice.Info.Software.ApiVersionPacked = props.apiVersion;
	OutDevice.Info.Software.DriverId = static_cast<OaU32>(driverProps.driverID);
	OutDevice.Info.Software.DriverName = OaString(driverProps.driverName);
	OutDevice.Info.Software.DriverInfo = OaString(driverProps.driverInfo);
	
	OutDevice.Info.Software.HasCooperativeMatrix = FeatureBundle_.HasCoopMatrix;
	OutDevice.Info.Software.HasCooperativeVector = FeatureBundle_.HasCoopVector;
	OutDevice.Info.Software.HasCooperativeMatrix2 = FeatureBundle_.HasCoopMatrix2;
	OutDevice.Info.Software.HasCooperativeMatrixDecodeVector = FeatureBundle_.HasCoopMatrixDecodeVector;
	OutDevice.Info.Software.CoopMatShapes = FeatureBundle_.DiscoveredCoopMatShapes;
	OutDevice.Info.Software.HasPipelineLibrary = ExtProbe_.PipelineLibrary;
	OutDevice.Info.Software.Has16BitStorage = FeatureBundle_.Has16bit;
	OutDevice.Info.Software.ShaderFloat16Enabled = FeatureBundle_.Has16bit && (FeatureBundle_.Supported12.shaderFloat16 == VK_TRUE);
	OutDevice.Info.Software.ShaderBfloat16ExtensionEnabled = FeatureBundle_.WantEnableBf16Ext;
	OutDevice.Info.Software.ShaderBfloat16TypeEnabled = FeatureBundle_.WantEnableBf16Ext && (FeatureBundle_.EnableBf16Feat.shaderBFloat16Type == VK_TRUE);
	OutDevice.Info.Software.ShaderBfloat16DotProductEnabled = FeatureBundle_.WantEnableBf16Ext && (FeatureBundle_.EnableBf16Feat.shaderBFloat16DotProduct == VK_TRUE);
	OutDevice.Info.Software.ShaderBfloat16CooperativeMatrixEnabled = FeatureBundle_.WantEnableBf16Ext && (FeatureBundle_.EnableBf16Feat.shaderBFloat16CooperativeMatrix == VK_TRUE);
	OutDevice.Info.Software.ShaderIntegerDotProductEnabled = FeatureBundle_.HasIntDotProduct && (FeatureBundle_.Features13.shaderIntegerDotProduct == VK_TRUE);
	OutDevice.Info.Software.HasDeviceGeneratedCommands = FeatureBundle_.HasDeviceGeneratedCommands;
	OutDevice.Info.Software.HasSwapchainMaintenance1 = FeatureBundle_.HasSwapchainMaintenance1;
	
	// Apply vendor/arch/driver trust gate to CoopMat (mirrors llama.cpp)
	if (OutDevice.Info.Software.HasCooperativeMatrix) {
		const bool userDisable = OaEnvFlag::IsSet("OA_DISABLE_COOPMAT");
		bool untrusted = false;
		const char* reason = "";
		if (userDisable) {
			untrusted = true;
			reason = "disabled by OA_DISABLE_COOPMAT=1";
		} else {
			const bool trusted = OaCoopMatTrust(
				OutDevice.Info.Hardware.VendorId,
				OutDevice.Info.Hardware.DeviceId,
				OutDevice.Info.Software.DriverId);
			if (!trusted) {
				untrusted = true;
				reason = "vendor/driver blacklisted (override with OA_FORCE_COOPMAT=1)";
			}
		}
		if (untrusted) {
			OA_LOG_WARN(OaLogComponent::Core,
				"CoopMat: %s — falling back to scalar paths "
				"(vendor=0x%04X device=0x%04X driverId=%u)",
				reason,
				OutDevice.Info.Hardware.VendorId,
				OutDevice.Info.Hardware.DeviceId,
				OutDevice.Info.Software.DriverId);
			OutDevice.Info.Software.HasCooperativeMatrix = false;
			OutDevice.Info.Software.ShaderBfloat16CooperativeMatrixEnabled = false;
		}
	}

	// Apply vendor/arch/driver trust gate to native BF16 (mirrors the CoopMat gate
	// above). A driver may advertise shaderBFloat16Type but miscompile bf16 arithmetic
	// (Intel pre-Xe2 Mesa/ANV, AMD pre-RDNA3 blob) -> silently wrong training, not a
	// crash. Withhold native bf16 unless trusted; OA_FORCE_BF16=1 overrides and
	// OA_DISABLE_BF16=1 forces it off. Routing then uses the FP32 storage path.
	if (OutDevice.Info.Software.ShaderBfloat16TypeEnabled) {
		const bool userDisable = OaEnvFlag::IsSet("OA_DISABLE_BF16");
		bool untrusted = userDisable;
		const char* reason = userDisable ? "disabled by OA_DISABLE_BF16=1" : "";
		if (not untrusted and not OaBf16Trust(
				OutDevice.Info.Hardware.VendorId,
				OutDevice.Info.Hardware.DeviceId,
				OutDevice.Info.Software.DriverId)) {
			untrusted = true;
			reason = "vendor/driver blacklisted for native bf16 (override with OA_FORCE_BF16=1)";
		}
		if (untrusted) {
			OA_LOG_WARN(OaLogComponent::Core,
				"BF16: %s — using FP32 (vendor=0x%04X device=0x%04X driverId=%u)",
				reason,
				OutDevice.Info.Hardware.VendorId,
				OutDevice.Info.Hardware.DeviceId,
				OutDevice.Info.Software.DriverId);
			OutDevice.Info.Software.ShaderBfloat16TypeEnabled = false;
			OutDevice.Info.Software.ShaderBfloat16ExtensionEnabled = false;
			OutDevice.Info.Software.ShaderBfloat16DotProductEnabled = false;
			OutDevice.Info.Software.ShaderBfloat16CooperativeMatrixEnabled = false;
		}
	}
	
	OutDevice.Info.Software.HasExternalMemoryFd = ExtProbe_.ExternalMemoryFd;
	
	// Video extensions
	OutDevice.Info.Software.HasVideoQueue = ExtProbe_.KhrVideoQueue;
	OutDevice.Info.Software.HasVideoDecodeQueue = ExtProbe_.KhrVideoDecodeQueue;
	OutDevice.Info.Software.HasVideoDecodeH264 = ExtProbe_.KhrVideoDecodeH264;
	OutDevice.Info.Software.HasVideoDecodeH265 = ExtProbe_.KhrVideoDecodeH265;
	OutDevice.Info.Software.HasVideoDecodeAV1 = ExtProbe_.KhrVideoDecodeAV1;
	OutDevice.Info.Software.HasVideoDecodeVP9 = ExtProbe_.KhrVideoDecodeVP9 && FeatureBundle_.HasVideoDecodeVp9;
	OutDevice.Info.Software.HasVideoEncodeQueue = ExtProbe_.KhrVideoEncodeQueue;
	OutDevice.Info.Software.HasVideoEncodeH264 = ExtProbe_.KhrVideoEncodeH264;
	OutDevice.Info.Software.HasVideoEncodeH265 = ExtProbe_.KhrVideoEncodeH265;
	OutDevice.Info.Software.HasVideoEncodeAV1 = ExtProbe_.KhrVideoEncodeAV1;
	OutDevice.Info.Software.HasSamplerYcbcrConversion = ExtProbe_.KhrSamplerYcbcr;

	// Copy enabled extensions
	for (const char* ext : EnabledExtensions_) {
		OutDevice.Info.Software.EnabledDeviceExtensions.PushBack(ext);
	}
}
