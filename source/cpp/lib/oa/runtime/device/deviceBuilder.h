// OA vulkan Device Builder
// type-safe builder for creating devices with specific feature sets
#pragma once

#include <oa/core/types.h>
#include <oa/core/status.h>
#include <oa/runtime/device.h>
#include "featureModule.h"

namespace oavk {

[[nodiscard]] oa::Status planDeviceQueues(
	const OaVkInstanceTable& inDispatch,
	VkPhysicalDevice inPhysicalDevice,
	VkSurfaceKHR inSurface,
	QueuePlan& outPlan,
	bool inNeedsGraphics
);


// ─────────────────────────────────────────────────────────────────────────────
// DeviceBuilder — type-safe device builder with modular features
//
// usage:
//   auto device = DeviceBuilder()
//       .withCore()
//       .withMl()
//       .buildCompute(instance, physicalDevice);
// ─────────────────────────────────────────────────────────────────────────────
class DeviceBuilder {
public:
	DeviceBuilder() = default;

	// ─── phase 1: Module Registration ───
	DeviceBuilder& withCore();
	DeviceBuilder& withMl();
	DeviceBuilder& withVision();
	DeviceBuilder& withAudio();
	DeviceBuilder& withRender();

	// Convenience methods
	DeviceBuilder& withAllCompute();  // Core + ML + Vision + Audio
	DeviceBuilder& withAllFeatures(); // Core + ML + Vision + Audio + Render

	// ─── phase 2-7: Internal Build pipeline ───
	// These are called internally by Build*() methods

	// phase 2: probe extensions
	void probeExtensions(const oa::Vec<VkExtensionProperties>& inExtensions);

	// phase 3: query features
	void queryFeatures(
		const OaVkInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice);

	// phase 4: Build feature chains
	void buildFeatureChains();

	// phase 5: Collect extensions
	void collectExtensions();

	// ─── phase 8: type-Safe Device Creation ───
	// inWantsPresentation enables surface-dependent device features for a later
	// oa::Presenter attachment. inNeedsGraphics selects a graphics-capable queue
	// independently for headless rendering. Both expose graphicsQueue and
	// graphicsQueueFamily; only the first enables WSI extensions.
	[[nodiscard]] oa::Result<oavk::Device> buildBase(
		VkInstance inInstance,
		VkPhysicalDevice inPhysicalDevice,
		oa::Bool inEnableValidation = false,
		oa::Bool inWantsPresentation = false,
		oa::Bool inNeedsGraphics = false
	);

	[[nodiscard]] oa::Result<oavk::ComputeDevice> buildCompute(
		VkInstance inInstance,
		VkPhysicalDevice inPhysicalDevice,
		oa::Bool inEnableValidation = false
	);

	[[nodiscard]] oa::Result<oavk::RenderDevice> buildRender(
		VkInstance inInstance,
		VkPhysicalDevice inPhysicalDevice,
		oa::Bool inEnableValidation = false,
		VkSurfaceKHR inSurface = VK_NULL_HANDLE
	);

private:
	// Module storage
	oa::Vec<oa::UniquePtr<FeatureModule>> modules_;
	
	// Module flags
	bool hasCoreModule_    = false;
	bool hasMlModule_      = false;
	bool hasVisionModule_  = false;
	bool hasAudioModule_   = false;
	bool hasRenderModule_  = false;

	// Build state
	PhysicalExtensionProbe extProbe_;
	DeviceFeatureBundle featureBundle_;
	oa::Vec<const char*> enabledExtensions_;

	// Helper: validate module dependencies
	oa::Status validateDependencies() const;

	// Helper: Sort modules by dependencies
	void sortModulesByDependencies();

	// Helper: Create logical device
	oa::Result<VkDevice> createLogicalDevice(
		const OaVkInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		const QueuePlan& inQueuePlan
	);

	// Helper: Populate device info
	void populateDeviceInfo(
		VkPhysicalDevice inPhysicalDevice,
		VkDevice inDevice,
		oavk::Device& outDevice
	);
};

} // namespace oavk
