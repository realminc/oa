// OA vulkan Device Builder Implementation
#include "deviceBuilder.h"
#include <oa/core/log.h>
#include <oa/core/envFlag.h>
#include <oa/runtime/init.h>
#include <oa/runtime/bindless.h>
#include <oa/core/std/algo.h>

// forward declarations from Device.cpp
oa::Status oavk::planDeviceQueues(
	const OaVkInstanceTable& inDispatch,
	VkPhysicalDevice inPhys,
	VkSurfaceKHR inSurface,
	oavk::QueuePlan& outPlan,
	bool inNeedsGraphics
);


// ─────────────────────────────────────────────────────────────────────────────
// phase 1: Module Registration
// ─────────────────────────────────────────────────────────────────────────────

oavk::DeviceBuilder& oavk::DeviceBuilder::withCore() {
	if (!hasCoreModule_) {
		modules_.pushBack(oavk::createCoreFeatures());
		hasCoreModule_ = true;
	}
	return *this;
}


oavk::DeviceBuilder& oavk::DeviceBuilder::withMl() {
	if (!hasMlModule_) {
		modules_.pushBack(oavk::createMlFeatures());
		hasMlModule_ = true;
	}
	return *this;
}


oavk::DeviceBuilder& oavk::DeviceBuilder::withVision() {
	if (!hasVisionModule_) {
		modules_.pushBack(oavk::createVisionFeatures());
		hasVisionModule_ = true;
	}
	return *this;
}


oavk::DeviceBuilder& oavk::DeviceBuilder::withAudio() {
	if (!hasAudioModule_) {
		modules_.pushBack(oavk::createAudioFeatures());
		hasAudioModule_ = true;
	}
	return *this;
}


oavk::DeviceBuilder& oavk::DeviceBuilder::withRender() {
	if (!hasRenderModule_) {
		modules_.pushBack(oavk::createRenderFeatures());
		hasRenderModule_ = true;
	}
	return *this;
}


oavk::DeviceBuilder& oavk::DeviceBuilder::withAllCompute() {
	return withCore().withMl().withVision().withAudio();
}


oavk::DeviceBuilder& oavk::DeviceBuilder::withAllFeatures() {
	return withCore().withMl().withVision().withAudio().withRender();
}


// ─────────────────────────────────────────────────────────────────────────────
// phase 2: probe Extensions
// ─────────────────────────────────────────────────────────────────────────────

void oavk::DeviceBuilder::probeExtensions(const oa::Vec<VkExtensionProperties>& inExtensions) {
	for (const auto& module : modules_) {
		module->probeExtensions(inExtensions, extProbe_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// phase 3: query features
// ─────────────────────────────────────────────────────────────────────────────

void oavk::DeviceBuilder::queryFeatures(
	const OaVkInstanceTable& inDispatch,
	VkPhysicalDevice inPhysicalDevice)
{
	for (const auto& module : modules_) {
		module->queryFeatures(inDispatch, inPhysicalDevice, featureBundle_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// phase 4: Build Feature Chains
// ─────────────────────────────────────────────────────────────────────────────

void oavk::DeviceBuilder::buildFeatureChains() {
	for (const auto& module : modules_) {
		module->buildFeatureChain(featureBundle_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// phase 5: Collect Extensions
// ─────────────────────────────────────────────────────────────────────────────

void oavk::DeviceBuilder::collectExtensions() {
	enabledExtensions_.clear();
	for (const auto& module : modules_) {
		module->collectExtensions(extProbe_, featureBundle_, enabledExtensions_);
	}
}


// ─────────────────────────────────────────────────────────────────────────────
// Dependency Validation
// ─────────────────────────────────────────────────────────────────────────────

oa::Status oavk::DeviceBuilder::validateDependencies() const {
	// Build module name map
	oa::HashSet<oa::StringView> availableModules;
	for (const auto& module : modules_) {
		availableModules.insert(module->name());
	}

	// Check each module's dependencies
	for (const auto& module : modules_) {
		const auto deps = module->dependencies();
		for (const auto& dep : deps) {
			if (!availableModules.contains(dep)) {
				return oa::Status::error(
					oa::StatusCode::FailedPrecondition,
					oa::String("Module '") + oa::String(module->name()) +
					"' requires module '" + oa::String(dep) + "' which is not loaded"
				);
			}
		}
	}

	return oa::Status::ok();
}


void oavk::DeviceBuilder::sortModulesByDependencies() {
	// Simple topological sort: move modules with no dependencies first
	// This is sufficient for our small module count
	oa::stableSort(modules_.begin(), modules_.end(),
		[](const oa::UniquePtr<oavk::FeatureModule>& a, const oa::UniquePtr<oavk::FeatureModule>& b) {
			return a->dependencies().size() < b->dependencies().size();
		}
	);
}


// ─────────────────────────────────────────────────────────────────────────────
// phase 8: type-Safe Device Creation
// ─────────────────────────────────────────────────────────────────────────────

oa::Result<oavk::Device> oavk::DeviceBuilder::buildBase(
	VkInstance inInstance,
	VkPhysicalDevice inPhysicalDevice,
	oa::Bool inEnableValidation,
	oa::Bool inWantsPresentation,
	oa::Bool inNeedsGraphics
) {
	// validate Core module is present
	if (!hasCoreModule_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"buildBase requires withCore()");
	}

	// validate and sort modules
	oa::Status depStatus = validateDependencies();
	if (!depStatus.isOk()) {
		return depStatus;
	}
	sortModulesByDependencies();
	OaVkInstanceTable instanceDispatch{};
	oaVkLoadInstanceTable(&instanceDispatch, inInstance);
	if (instanceDispatch.vkEnumerateDeviceExtensionProperties == nullptr
		or instanceDispatch.vkCreateDevice == nullptr)
	{
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"vulkan instance dispatch is incomplete");
	}

	// Enumerate extensions
	oa::U32 extCount = 0;
	instanceDispatch.vkEnumerateDeviceExtensionProperties(
		inPhysicalDevice, nullptr, &extCount, nullptr);
	oa::Vec<VkExtensionProperties> extensions(extCount);
	instanceDispatch.vkEnumerateDeviceExtensionProperties(
		inPhysicalDevice, nullptr, &extCount, extensions.data());

	// run build pipeline
	probeExtensions(extensions);
	queryFeatures(instanceDispatch, inPhysicalDevice);
	if (!inWantsPresentation) {
		featureBundle_.hasSwapchainMaintenance1 = false;
	}
	buildFeatureChains();
	collectExtensions();
	// withAllFeatures() is also used by the headless compute context.  Merely
	// loading the render feature module must not enable VK_KHR_swapchain: the
	// extension requires the instance-side surface extension even when no
	// swapchain is ever created (VUID-vkCreateDevice-ppEnabledExtensionNames-01387).
	if (!inWantsPresentation) {
		for (oa::Usize i = 0; i < enabledExtensions_.size();) {
			if (oa::strcmp(enabledExtensions_[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
				enabledExtensions_.erase(enabledExtensions_.begin() + i);
			} else {
				++i;
			}
		}
	}

	// Plan queues. When the caller hinted at later swapchain attachment,
	// the planner picks a graphics-capable family even though inSurface is
	// null here (verified against the real surface during initPresentation).
	oavk::QueuePlan queuePlan;
	oa::Status queueStatus = oavk::planDeviceQueues(
		instanceDispatch,
		inPhysicalDevice, VK_NULL_HANDLE, queuePlan,
		inWantsPresentation or inNeedsGraphics);
	if (!queueStatus.isOk()) {
		return queueStatus;
	}

	// Create logical device
	auto deviceResult = createLogicalDevice(
		instanceDispatch, inPhysicalDevice, queuePlan);
	if (!deviceResult.isOk()) {
		return deviceResult.getStatus();
	}

	// Populate device structure
	oavk::Device device;
	device.instance = inInstance;
	device.physicalDevice = inPhysicalDevice;
	device.device = deviceResult.getValue();
	device.ownsInstance = false;
	device.instanceDispatch = instanceDispatch;
	oaVkLoadDeviceTable(
		&device.deviceDispatch,
		&device.instanceDispatch,
		static_cast<VkDevice>(device.device));

	// Retrieve queue handles
	VkDevice vkDevice = deviceResult.getValue();
	VkQueue computeQ = VK_NULL_HANDLE;
	VkQueue asyncComputeQ = VK_NULL_HANDLE;
	VkQueue transferQ = VK_NULL_HANDLE;

	device.deviceDispatch.vkGetDeviceQueue(vkDevice, queuePlan.computeQF, 0, &computeQ);
	device.deviceDispatch.vkGetDeviceQueue(vkDevice, queuePlan.transferQF, 0, &transferQ);

	oa::U32 actualAsyncQF = queuePlan.computeQF;
	if (queuePlan.asyncComputeQF != UINT32_MAX) {
		device.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.asyncComputeQF, 0, &asyncComputeQ);
		actualAsyncQF = queuePlan.asyncComputeQF;
	} else if (queuePlan.computeHasMultiQueue) {
		device.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.computeQF, 1, &asyncComputeQ);
		actualAsyncQF = queuePlan.computeQF;
	}

	// Populate queue info
	device.queues.computeQueue = computeQ;
	device.queues.asyncComputeQueue = asyncComputeQ;
	device.queues.transferQueue = transferQ;
	device.queues.computeQueueFamily = queuePlan.computeQF;
	device.queues.asyncComputeQueueFamily = actualAsyncQF;
	device.queues.transferQueueFamily = queuePlan.transferQF;
	device.queues.hasAsyncCompute = queuePlan.hasAsync;
	device.queues.computeQueueSlotCount = queuePlan.computeSlots;
	device.queues.dedicatedTransferQueueSlotCount = queuePlan.dedicatedTransferSlots;

	// step 3g.1.b followup: when the planner found video-decode / video-encode
	// queue families on the device, fetch their VkQueue handles now and
	// publish them on device.queues so oa::VideoDecoder / oa::VideoEncoder can
	// submit on the right queue. Without this, the planner would bump the
	// queueCount for these families in the VkDeviceCreateInfo but the queue
	// handles were never extracted — IsCodecSupported returned false even on
	// devices that genuinely exposed the extensions.
	if (queuePlan.videoDecodeQF != UINT32_MAX) {
		VkQueue videoDecodeQ = VK_NULL_HANDLE;
		device.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.videoDecodeQF, 0, &videoDecodeQ);
		device.queues.videoDecodeQueue        = videoDecodeQ;
		device.queues.videoDecodeQueueFamily  = queuePlan.videoDecodeQF;
		device.queues.hasVideoDecodeQueue     = (videoDecodeQ != VK_NULL_HANDLE);
		device.queues.videoDecodeCodecOps     = queuePlan.videoDecodeCodecOps;
	}
	if (queuePlan.videoEncodeQF != UINT32_MAX) {
		VkQueue videoEncodeQ = VK_NULL_HANDLE;
		device.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.videoEncodeQF, 0, &videoEncodeQ);
		device.queues.videoEncodeQueue        = videoEncodeQ;
		device.queues.videoEncodeQueueFamily  = queuePlan.videoEncodeQF;
		device.queues.hasVideoEncodeQueue     = (videoEncodeQ != VK_NULL_HANDLE);
		device.queues.videoEncodeCodecOps     = queuePlan.videoEncodeCodecOps;
	}

	// A graphics queue is useful independently of WSI. Presentation queue state
	// remains unset until oa::Presenter verifies the actual surface.
	if (queuePlan.wantsGraphics
		and queuePlan.graphicsQF != UINT32_MAX) {
		VkQueue graphicsQ = VK_NULL_HANDLE;
		device.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.graphicsQF, 0, &graphicsQ);
		device.queues.graphicsQueue        = graphicsQ;
		device.queues.graphicsQueueFamily  = queuePlan.graphicsQF;
	}

	populateDeviceInfo(inPhysicalDevice, deviceResult.getValue(), device);

	return device;
}


oa::Result<oavk::ComputeDevice> oavk::DeviceBuilder::buildCompute(
	VkInstance inInstance,
	VkPhysicalDevice inPhysicalDevice,
	oa::Bool inEnableValidation
) {
	// Build base device first
	auto baseResult = buildBase(inInstance, inPhysicalDevice, inEnableValidation);
	if (!baseResult.isOk()) {
		return baseResult.getStatus();
	}

	// Upcast to compute device
	oavk::ComputeDevice computeDevice;
	static_cast<oavk::Device&>(computeDevice) = baseResult.getValue();

	// Populate compute-class fields from Info.software (already trust-gated by
	// buildBase). Previously this block read directly from featureBundle_,
	// which bypassed the vendor trust gate — e.g. on an untrusted device,
	// Info.software.hasCooperativeMatrix would be false but
	// computeDevice.hasCooperativeMatrix would silently stay true.
	computeDevice.syncFromSoftwareInfo();
	computeDevice.logCoopMatShapes();

	return computeDevice;
}


oa::Result<oavk::RenderDevice> oavk::DeviceBuilder::buildRender(
	VkInstance inInstance,
	VkPhysicalDevice inPhysicalDevice,
	oa::Bool inEnableValidation,
	VkSurfaceKHR inSurface
) {
	// validate Render module is present
	if (!hasRenderModule_) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition, "BuildRender requires withRender()");
	}

	// validate and sort modules
	oa::Status depStatus = validateDependencies();
	if (!depStatus.isOk()) {
		return depStatus;
	}
	sortModulesByDependencies();
	OaVkInstanceTable instanceDispatch{};
	oaVkLoadInstanceTable(&instanceDispatch, inInstance);
	if (instanceDispatch.vkEnumerateDeviceExtensionProperties == nullptr
		or instanceDispatch.vkCreateDevice == nullptr)
	{
		return oa::Status::error(
			oa::StatusCode::FailedPrecondition,
			"vulkan instance dispatch is incomplete");
	}

	// Enumerate extensions
	oa::U32 extCount = 0;
	instanceDispatch.vkEnumerateDeviceExtensionProperties(
		inPhysicalDevice, nullptr, &extCount, nullptr);
	oa::Vec<VkExtensionProperties> extensions(extCount);
	instanceDispatch.vkEnumerateDeviceExtensionProperties(
		inPhysicalDevice, nullptr, &extCount, extensions.data());

	// run build pipeline
	probeExtensions(extensions);
	queryFeatures(instanceDispatch, inPhysicalDevice);
	if (inSurface == VK_NULL_HANDLE) {
		featureBundle_.hasSwapchainMaintenance1 = false;
	}
	buildFeatureChains();
	collectExtensions();
	if (inSurface == VK_NULL_HANDLE) {
		for (oa::Usize i = 0; i < enabledExtensions_.size();) {
			if (oa::strcmp(enabledExtensions_[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
				enabledExtensions_.erase(enabledExtensions_.begin() + i);
			} else {
				++i;
			}
		}
	}

	// Plan queues with graphics/present support
	oavk::QueuePlan queuePlan;
	bool wantsPresentation = (inSurface != VK_NULL_HANDLE);
	oa::Status queueStatus = oavk::planDeviceQueues(
		instanceDispatch,
		inPhysicalDevice, inSurface, queuePlan, wantsPresentation);
	if (!queueStatus.isOk()) {
		return queueStatus;
	}

	// Create logical device
	auto deviceResult = createLogicalDevice(
		instanceDispatch, inPhysicalDevice, queuePlan);
	if (!deviceResult.isOk()) {
		return deviceResult.getStatus();
	}

	// Build base device structure
	oavk::RenderDevice renderDevice;
	renderDevice.instance = inInstance;
	renderDevice.physicalDevice = inPhysicalDevice;
	renderDevice.device = deviceResult.getValue();
	renderDevice.ownsInstance = false;
	renderDevice.instanceDispatch = instanceDispatch;
	oaVkLoadDeviceTable(
		&renderDevice.deviceDispatch,
		&renderDevice.instanceDispatch,
		static_cast<VkDevice>(renderDevice.device));

	// Retrieve queue handles (compute + graphics/present)
	VkDevice vkDevice = deviceResult.getValue();
	VkQueue computeQ = VK_NULL_HANDLE;
	VkQueue asyncComputeQ = VK_NULL_HANDLE;
	VkQueue transferQ = VK_NULL_HANDLE;
	VkQueue graphicsQ = VK_NULL_HANDLE;
	VkQueue presentQ = VK_NULL_HANDLE;
	
	renderDevice.deviceDispatch.vkGetDeviceQueue(vkDevice, queuePlan.computeQF, 0, &computeQ);
	renderDevice.deviceDispatch.vkGetDeviceQueue(vkDevice, queuePlan.transferQF, 0, &transferQ);
	
	oa::U32 actualAsyncQF = queuePlan.computeQF;
	if (queuePlan.asyncComputeQF != UINT32_MAX) {
		renderDevice.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.asyncComputeQF, 0, &asyncComputeQ);
		actualAsyncQF = queuePlan.asyncComputeQF;
	} else if (queuePlan.computeHasMultiQueue) {
		renderDevice.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.computeQF, 1, &asyncComputeQ);
		actualAsyncQF = queuePlan.computeQF;
	}
	
	if (queuePlan.wantsGraphics) {
		renderDevice.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.graphicsQF, 0, &graphicsQ);
		if (inSurface != VK_NULL_HANDLE) presentQ = graphicsQ;
	}
	
	// Populate queue info
	renderDevice.queues.computeQueue = computeQ;
	renderDevice.queues.asyncComputeQueue = asyncComputeQ;
	renderDevice.queues.transferQueue = transferQ;
	renderDevice.queues.graphicsQueue = graphicsQ;
	renderDevice.queues.presentQueue = presentQ;
	renderDevice.queues.computeQueueFamily = queuePlan.computeQF;
	renderDevice.queues.asyncComputeQueueFamily = actualAsyncQF;
	renderDevice.queues.transferQueueFamily = queuePlan.transferQF;
	renderDevice.queues.graphicsQueueFamily = queuePlan.graphicsQF;
	renderDevice.queues.presentQueueFamily = queuePlan.presentQF;
	renderDevice.queues.hasAsyncCompute = queuePlan.hasAsync;
	renderDevice.queues.hasPresentation = (inSurface != VK_NULL_HANDLE) && (graphicsQ != VK_NULL_HANDLE);
	renderDevice.queues.computeQueueSlotCount = queuePlan.computeSlots;
	renderDevice.queues.dedicatedTransferQueueSlotCount = queuePlan.dedicatedTransferSlots;

	// Same video-queue fix as buildBase — fetch the video-decode/encode
	// VkQueue handles when the planner found their families.
	if (queuePlan.videoDecodeQF != UINT32_MAX) {
		VkQueue videoDecodeQ = VK_NULL_HANDLE;
		renderDevice.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.videoDecodeQF, 0, &videoDecodeQ);
		renderDevice.queues.videoDecodeQueue        = videoDecodeQ;
		renderDevice.queues.videoDecodeQueueFamily  = queuePlan.videoDecodeQF;
		renderDevice.queues.hasVideoDecodeQueue     = (videoDecodeQ != VK_NULL_HANDLE);
		renderDevice.queues.videoDecodeCodecOps     = queuePlan.videoDecodeCodecOps;
	}
	if (queuePlan.videoEncodeQF != UINT32_MAX) {
		VkQueue videoEncodeQ = VK_NULL_HANDLE;
		renderDevice.deviceDispatch.vkGetDeviceQueue(
			vkDevice, queuePlan.videoEncodeQF, 0, &videoEncodeQ);
		renderDevice.queues.videoEncodeQueue        = videoEncodeQ;
		renderDevice.queues.videoEncodeQueueFamily  = queuePlan.videoEncodeQF;
		renderDevice.queues.hasVideoEncodeQueue     = (videoEncodeQ != VK_NULL_HANDLE);
		renderDevice.queues.videoEncodeCodecOps     = queuePlan.videoEncodeCodecOps;
	}

	// Populate device info (also applies the CoopMat vendor trust gate to
	// renderDevice.info.software — see PopulateDeviceInfo body).
	populateDeviceInfo(inPhysicalDevice, vkDevice, static_cast<oavk::Device&>(renderDevice));

	// Mirror trust-gated Info.software.* into class-level fields. Same fix as
	// buildCompute — previously read from featureBundle_ which bypasses the
	// gate.
	renderDevice.syncFromSoftwareInfo();

	// Populate render-specific capabilities
	renderDevice.hasGraphicsQueue    = graphicsQ != VK_NULL_HANDLE;
	renderDevice.hasPresentQueue     = presentQ  != VK_NULL_HANDLE;
	renderDevice.hasSwapchainSupport = extProbe_.khrSwapchain;
	renderDevice.info.software.hasSwapchainMaintenance1 =
		featureBundle_.hasSwapchainMaintenance1 && inSurface != VK_NULL_HANDLE;

	return renderDevice;
}


// ─────────────────────────────────────────────────────────────────────────────
// Helper: Create Logical Device
// ─────────────────────────────────────────────────────────────────────────────

oa::Result<VkDevice> oavk::DeviceBuilder::createLogicalDevice(
	const OaVkInstanceTable& inDispatch,
	VkPhysicalDevice inPhysicalDevice,
	const oavk::QueuePlan& inQueuePlan
) {
	VkDeviceCreateInfo devCI{};
	devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	devCI.pNext = &featureBundle_.features2;
	devCI.queueCreateInfoCount = static_cast<oa::U32>(inQueuePlan.queueCIs.size());
	devCI.pQueueCreateInfos = inQueuePlan.queueCIs.data();
	devCI.enabledExtensionCount = static_cast<oa::U32>(enabledExtensions_.size());
	devCI.ppEnabledExtensionNames = enabledExtensions_.data();

	VkDevice device = VK_NULL_HANDLE;
	VkResult r = inDispatch.vkCreateDevice(
		inPhysicalDevice, &devCI, nullptr, &device);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkCreateDevice failed");
	}

	return device;
}


// ─────────────────────────────────────────────────────────────────────────────
// Helper: Populate Device Info
// ─────────────────────────────────────────────────────────────────────────────

void oavk::DeviceBuilder::populateDeviceInfo(
	VkPhysicalDevice inPhysicalDevice,
	VkDevice inDevice,
	oavk::Device& outDevice
) {
	// get physical device properties
	VkPhysicalDeviceProperties props{};
	outDevice.instanceDispatch.vkGetPhysicalDeviceProperties(
		inPhysicalDevice, &props);

	// Populate hardware info
	outDevice.info.hardware.deviceName = props.deviceName;
	outDevice.info.hardware.vendorId = props.vendorID;
	outDevice.info.hardware.deviceId = props.deviceID;
	outDevice.info.hardware.deviceType = oavk::mapPhysicalType(props.deviceType);
	outDevice.info.hardware.vendorName = oavk::vendorLabel(props.vendorID);

	// get memory info
	VkPhysicalDeviceMemoryProperties memProps{};
	outDevice.instanceDispatch.vkGetPhysicalDeviceMemoryProperties(
		inPhysicalDevice, &memProps);
	oa::U64 vram = 0;
	for (oa::U32 i = 0; i < memProps.memoryHeapCount; i++) {
		if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			vram += memProps.memoryHeaps[i].size;
		}
	}
	outDevice.info.hardware.vramBytes = vram;

	// get subgroup size and driver info
	VkPhysicalDeviceSubgroupProperties subgroupProps{};
	subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	VkPhysicalDeviceDriverProperties driverProps{};
	driverProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
	subgroupProps.pNext = &driverProps;
	VkPhysicalDeviceProperties2 props2{};
	props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2.pNext = &subgroupProps;
	outDevice.instanceDispatch.vkGetPhysicalDeviceProperties2(
		inPhysicalDevice, &props2);
	
	outDevice.info.hardware.subgroupSize = subgroupProps.subgroupSize;
	outDevice.info.hardware.maxComputeWorkGroupSize = props.limits.maxComputeWorkGroupSize[0];
	outDevice.info.hardware.maxComputeWorkGroupInvocations = props.limits.maxComputeWorkGroupInvocations;
	outDevice.info.hardware.maxComputeWorkGroupCountX =
		props.limits.maxComputeWorkGroupCount[0];
	outDevice.info.hardware.maxComputeWorkGroupCountY =
		props.limits.maxComputeWorkGroupCount[1];
	outDevice.info.hardware.maxComputeWorkGroupCountZ =
		props.limits.maxComputeWorkGroupCount[2];
	outDevice.info.hardware.maxComputeSharedMemoryBytes = props.limits.maxComputeSharedMemorySize;
	outDevice.info.hardware.maxStorageBufferRangeBytes =
		static_cast<oa::U64>(props.limits.maxStorageBufferRange);
	outDevice.info.hardware.timestampPeriodNanoseconds =
		static_cast<oa::F64>(props.limits.timestampPeriod);
	oa::U32 queueFamilyCount = 0;
	outDevice.instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties(
		inPhysicalDevice, &queueFamilyCount, nullptr);
	oa::Vec<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	outDevice.instanceDispatch.vkGetPhysicalDeviceQueueFamilyProperties(
		inPhysicalDevice, &queueFamilyCount, queueFamilies.data());
	if (outDevice.queues.computeQueueFamily < queueFamilyCount) {
		outDevice.info.hardware.computeTimestampValidBits =
			queueFamilies[outDevice.queues.computeQueueFamily].timestampValidBits;
	}

	// query descriptor indexing limits for bindless heap capacity
	VkPhysicalDeviceDescriptorIndexingProperties indexingProps{};
	indexingProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
	VkPhysicalDeviceProperties2 props2ForIndexing{};
	props2ForIndexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2ForIndexing.pNext = &indexingProps;
	outDevice.instanceDispatch.vkGetPhysicalDeviceProperties2(
		inPhysicalDevice, &props2ForIndexing);

	// store the full reported limits for logging
	outDevice.info.hardware.maxPerStageDescriptorUpdateAfterBindStorageBuffers =
		static_cast<oa::U32>(indexingProps.maxPerStageDescriptorUpdateAfterBindStorageBuffers);
	outDevice.info.hardware.maxPerStageDescriptorUpdateAfterBindSampledImages =
		static_cast<oa::U32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages);
	outDevice.info.hardware.maxPerStageDescriptorUpdateAfterBindSamplers =
		static_cast<oa::U32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSamplers);

	// set bindless capacity based on device limits with safe caps.
	//
	// buffers: iGPUs report 100M+ theoretical UAB storage-buffer slots but cannot
	// back them efficiently. The descriptor pool AND our slot free-list are sized to
	// capacity, so an unclamped request means multi-second/minute init plus OOM on a
	// shared-memory ultrabook. Discrete GPUs are naturally bounded near 1M, so we cap
	// there; integrated / CPU devices get a tighter 256K cap (still vastly more live
	// tensors than any single ML graph needs on an iGPU). Either can be overridden at
	// runtime with OA_BINDLESS_BUFFER_CAP=N for debugging or large-model experiments.
	// images capped to 16K (pure compute) and samplers to 2K to avoid pipeline stalls.
	constexpr oa::U32 kSafeBufferLimitDiscrete   = 1048576;   // 1M
	constexpr oa::U32 kSafeBufferLimitIntegrated = 262144;    // 256K
	constexpr oa::U32 kSafeImageLimit            = 16384;     // 16K
	constexpr oa::U32 kSafeSamplerLimit          = 2048;      // 2K

	const bool isIntegrated =
		outDevice.info.hardware.deviceType == oa::DeviceType::VkIntegrated ||
		outDevice.info.hardware.deviceType == oa::DeviceType::VkCpu;
	oa::U32 bufferLimit = isIntegrated ? kSafeBufferLimitIntegrated : kSafeBufferLimitDiscrete;

	// Runtime override: OA_BINDLESS_BUFFER_CAP=N (later clamped to the reported limit).
	const oa::I64 envBufferCap = oa::EnvFlag::getInt("OA_BINDLESS_BUFFER_CAP", 0);
	if (envBufferCap > 0) {
		bufferLimit = static_cast<oa::U32>(envBufferCap);
		OaLogInfo(oa::LogComponent::Runtime,
			"bindless: buffer cap overridden via OA_BINDLESS_BUFFER_CAP=%lld",
			static_cast<long long>(envBufferCap));
	}

	outDevice.info.hardware.bindlessBufferCapacity = oa::min(
		static_cast<oa::U32>(indexingProps.maxPerStageDescriptorUpdateAfterBindStorageBuffers),
		bufferLimit
	);
	outDevice.info.hardware.bindlessImageCapacity = oa::min(
		static_cast<oa::U32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSampledImages),
		kSafeImageLimit
	);
	outDevice.info.hardware.bindlessSamplerCapacity = oa::min(
		static_cast<oa::U32>(indexingProps.maxPerStageDescriptorUpdateAfterBindSamplers),
		kSafeSamplerLimit
	);

	// Never inflate a requested descriptor count above the physical-device limit.
	// oavk::BindlessHeap reports Unavailable when a device cannot provide even the
	// reserved null slot plus one usable slot.

	// Detect SAM (Smart access memory / Resizable BAR)
	bool hasSam = false;
	for (oa::U32 i = 0; i < memProps.memoryTypeCount; i++) {
		if ((memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
			(memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
		{
			oa::U32 heapIdx = memProps.memoryTypes[i].heapIndex;
			if (memProps.memoryHeaps[heapIdx].size > 256ULL * 1024 * 1024) {
				hasSam = true;
				break;
			}
		}
	}
	outDevice.info.hardware.hasSAM = hasSam;
	
	// Estimate SM/CU count
	outDevice.info.hardware.numSMs = oavk::estimateNumSmsForDevice(
		props.vendorID, props.deviceID, outDevice.info.hardware.deviceType, vram);
	
	// Estimate performance metrics
	outDevice.info.hardware.estMemBandwidthGbps = oavk::estimateMemBandwidthGbpsForDevice(
		props.vendorID, props.deviceID, outDevice.info.hardware.deviceType, vram);
	outDevice.info.hardware.estPeakTflopsF32 = oavk::estimatePeakTflopsF32ForDevice(
		props.vendorID, props.deviceID, outDevice.info.hardware.deviceType, vram);
	
	// Populate software info
	outDevice.info.software.driverVersion = oavk::formatDriverVersion(props.driverVersion);
	outDevice.info.software.apiVersion = oavk::formatDriverVersion(props.apiVersion);
	outDevice.info.software.apiVersionPacked = props.apiVersion;
	outDevice.info.software.driverId = static_cast<oa::U32>(driverProps.driverID);
	outDevice.info.software.driverName = oa::String(driverProps.driverName);
	outDevice.info.software.driverInfo = oa::String(driverProps.driverInfo);
	
	outDevice.info.software.hasCooperativeMatrix = featureBundle_.hasCoopMatrix;
	outDevice.info.software.hasCooperativeVector = featureBundle_.hasCoopVector;
	outDevice.info.software.hasCooperativeMatrix2 = featureBundle_.hasCoopMatrix2;
	outDevice.info.software.hasCooperativeMatrixDecodeVector = featureBundle_.hasCoopMatrixDecodeVector;
	outDevice.info.software.coopMatShapes = featureBundle_.discoveredCoopMatShapes;
	outDevice.info.software.hasPipelineLibrary = extProbe_.pipelineLibrary;
	outDevice.info.software.has16BitStorage = featureBundle_.has16bit;
	outDevice.info.software.shaderFloat16Enabled = featureBundle_.has16bit && (featureBundle_.supported12.shaderFloat16 == VK_TRUE);
	outDevice.info.software.shaderFloat64Enabled =
		featureBundle_.features2.features.shaderFloat64 == VK_TRUE;
	outDevice.info.software.shaderBfloat16ExtensionEnabled = featureBundle_.wantEnableBf16Ext;
	outDevice.info.software.shaderBfloat16TypeEnabled = featureBundle_.wantEnableBf16Ext && (featureBundle_.enableBf16Feat.shaderBFloat16Type == VK_TRUE);
	outDevice.info.software.shaderBfloat16DotProductEnabled = featureBundle_.wantEnableBf16Ext && (featureBundle_.enableBf16Feat.shaderBFloat16DotProduct == VK_TRUE);
	outDevice.info.software.shaderBfloat16CooperativeMatrixEnabled = featureBundle_.wantEnableBf16Ext && (featureBundle_.enableBf16Feat.shaderBFloat16CooperativeMatrix == VK_TRUE);
	outDevice.info.software.shaderIntegerDotProductEnabled = featureBundle_.hasIntDotProduct && (featureBundle_.features13.shaderIntegerDotProduct == VK_TRUE);
	outDevice.info.software.hasDeviceGeneratedCommands = featureBundle_.hasDeviceGeneratedCommands;
	outDevice.info.software.hasSwapchainMaintenance1 = featureBundle_.hasSwapchainMaintenance1;
	
	// apply vendor/arch/driver trust gate to coopMat (mirrors llama.cpp)
	if (outDevice.info.software.hasCooperativeMatrix) {
		const bool userDisable = oa::EnvFlag::isSet("OA_DISABLE_COOPMAT");
		bool untrusted = false;
		const char* reason = "";
		if (userDisable) {
			untrusted = true;
			reason = "disabled by OA_DISABLE_COOPMAT=1";
		} else {
			const bool trusted = oavk::coopMatTrust(
				outDevice.info.hardware.vendorId,
				outDevice.info.hardware.deviceId,
				outDevice.info.software.driverId);
			if (!trusted) {
				untrusted = true;
				reason = "vendor/driver blacklisted (override with OA_FORCE_COOPMAT=1)";
			}
		}
		if (untrusted) {
			OaLogWarn(oa::LogComponent::Runtime,
				"CoopMat: %s — falling back to scalar paths "
				"(vendor=0x%04X device=0x%04X driverId=%u)",
				reason,
				outDevice.info.hardware.vendorId,
				outDevice.info.hardware.deviceId,
				outDevice.info.software.driverId);
			outDevice.info.software.hasCooperativeMatrix = false;
			outDevice.info.software.shaderBfloat16CooperativeMatrixEnabled = false;
		}
	}

	// apply vendor/arch/driver trust gate to native BF16 (mirrors the CoopMat gate
	// above). A driver may advertise shaderBFloat16Type but miscompile bf16 arithmetic
	// (Intel pre-Xe2 Mesa/ANV, AMD pre-RDNA3 blob) -> silently wrong training, not a
	// crash. Withhold native bf16 unless trusted; OA_FORCE_BF16=1 overrides and
	// OA_DISABLE_BF16=1 forces it off. Routing then uses the FP32 storage path.
	if (outDevice.info.software.shaderBfloat16TypeEnabled) {
		const bool userDisable = oa::EnvFlag::isSet("OA_DISABLE_BF16");
		bool untrusted = userDisable;
		const char* reason = userDisable ? "disabled by OA_DISABLE_BF16=1" : "";
		if (not untrusted and not oavk::bf16Trust(
				outDevice.info.hardware.vendorId,
				outDevice.info.hardware.deviceId,
				outDevice.info.software.driverId)) {
			untrusted = true;
			reason = "vendor/driver blacklisted for native bf16 (override with OA_FORCE_BF16=1)";
		}
		if (untrusted) {
			OaLogWarn(oa::LogComponent::Runtime,
				"BF16: %s — using FP32 (vendor=0x%04X device=0x%04X driverId=%u)",
				reason,
				outDevice.info.hardware.vendorId,
				outDevice.info.hardware.deviceId,
				outDevice.info.software.driverId);
			outDevice.info.software.shaderBfloat16TypeEnabled = false;
			outDevice.info.software.shaderBfloat16ExtensionEnabled = false;
			outDevice.info.software.shaderBfloat16DotProductEnabled = false;
			outDevice.info.software.shaderBfloat16CooperativeMatrixEnabled = false;
		}
	}
	
	outDevice.info.software.hasExternalMemoryFd = extProbe_.externalMemoryFd;
	outDevice.info.software.hasKhrCalibratedTimestamps =
		extProbe_.khrCalibratedTimestamps;
	outDevice.info.software.hasExtCalibratedTimestamps =
		!extProbe_.khrCalibratedTimestamps
		&& extProbe_.extCalibratedTimestamps;
	
	// Video extensions
	outDevice.info.software.hasVideoQueue = extProbe_.khrVideoQueue;
	outDevice.info.software.hasVideoDecodeQueue = extProbe_.khrVideoDecodeQueue;
	outDevice.info.software.hasVideoDecodeH264 = extProbe_.khrVideoDecodeH264;
	outDevice.info.software.hasVideoDecodeH265 = extProbe_.khrVideoDecodeH265;
	outDevice.info.software.hasVideoDecodeAV1 = extProbe_.khrVideoDecodeAV1;
	outDevice.info.software.hasVideoDecodeVP9 = extProbe_.khrVideoDecodeVP9 && featureBundle_.hasVideoDecodeVp9;
	outDevice.info.software.hasVideoEncodeQueue = extProbe_.khrVideoEncodeQueue;
	outDevice.info.software.hasVideoEncodeH264 = extProbe_.khrVideoEncodeH264;
	outDevice.info.software.hasVideoEncodeH265 = extProbe_.khrVideoEncodeH265;
	outDevice.info.software.hasVideoEncodeAV1 = extProbe_.khrVideoEncodeAV1;
	outDevice.info.software.hasSamplerYcbcrConversion = extProbe_.khrSamplerYcbcr;

	// Copy enabled extensions
	for (const char* ext : enabledExtensions_) {
		outDevice.info.software.enabledDeviceExtensions.pushBack(ext);
	}
}
