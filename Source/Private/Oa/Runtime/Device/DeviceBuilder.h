// OA Vulkan Device Builder
// Type-safe builder for creating devices with specific feature sets
#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Core/Status.h>
#include <Oa/Runtime/Device.h>
#include "FeatureModule.h"


// Forward declare factory functions
OaUniquePtr<OaVkFeatureModule> OaVkCreateCoreFeatures();
OaUniquePtr<OaVkFeatureModule> OaVkCreateMlFeatures();
OaUniquePtr<OaVkFeatureModule> OaVkCreateVisionFeatures();
OaUniquePtr<OaVkFeatureModule> OaVkCreateAudioFeatures();
OaUniquePtr<OaVkFeatureModule> OaVkCreateRenderFeatures();


// ─────────────────────────────────────────────────────────────────────────────
// OaVkDeviceBuilder — Type-safe device builder with modular features
//
// Usage:
//   auto device = OaVkDeviceBuilder()
//       .WithCore()
//       .WithMl()
//       .BuildCompute(instance, physicalDevice);
// ─────────────────────────────────────────────────────────────────────────────
class OaVkDeviceBuilder {
public:
	OaVkDeviceBuilder() = default;

	// ─── Phase 1: Module Registration ───
	OaVkDeviceBuilder& WithCore();
	OaVkDeviceBuilder& WithMl();
	OaVkDeviceBuilder& WithVision();
	OaVkDeviceBuilder& WithAudio();
	OaVkDeviceBuilder& WithRender();

	// Convenience methods
	OaVkDeviceBuilder& WithAllCompute();  // Core + ML + Vision + Audio
	OaVkDeviceBuilder& WithAllFeatures(); // Core + ML + Vision + Audio + Render

	// ─── Phase 2-7: Internal Build Pipeline ───
	// These are called internally by Build*() methods

	// Phase 2: Probe extensions
	void ProbeExtensions(const OaVec<VkExtensionProperties>& InExtensions);

	// Phase 3: Query features
	void QueryFeatures(VkPhysicalDevice InPhysicalDevice);

	// Phase 4: Build feature chains
	void BuildFeatureChains();

	// Phase 5: Collect extensions
	void CollectExtensions();

	// ─── Phase 8: Type-Safe Device Creation ───
	// InWantsPresentation: caller intends to attach a VkSurfaceKHR later (e.g.
	// via OaGraphicsEngine::InitPresentation). When true, the queue planner
	// selects a graphics-capable family without needing the surface at device-
	// creation time, and the resulting device exposes Queues.GraphicsQueue +
	// GraphicsQueueFamily. Surface-side presentation support is verified later
	// when the surface is actually attached. See UnifiedExecutionArchitecture.md §3.5.
	[[nodiscard]] OaResult<OaVkDevice> BuildBase(
		VkInstance InInstance,
		VkPhysicalDevice InPhysicalDevice,
		OaBool InEnableValidation = false,
		OaBool InWantsPresentation = false
	);

	[[nodiscard]] OaResult<OaVkComputeDevice> BuildCompute(
		VkInstance InInstance,
		VkPhysicalDevice InPhysicalDevice,
		OaBool InEnableValidation = false
	);

	[[nodiscard]] OaResult<OaVkRenderDevice> BuildRender(
		VkInstance InInstance,
		VkPhysicalDevice InPhysicalDevice,
		OaBool InEnableValidation = false,
		VkSurfaceKHR InSurface = VK_NULL_HANDLE
	);

private:
	// Module storage
	OaVec<OaUniquePtr<OaVkFeatureModule>> Modules_;
	
	// Module flags
	bool HasCoreModule_    = false;
	bool HasMlModule_      = false;
	bool HasVisionModule_  = false;
	bool HasAudioModule_   = false;
	bool HasRenderModule_  = false;

	// Build state
	OaVkPhysExtProbe ExtProbe_;
	OaVkDeviceFeatureBundle FeatureBundle_;
	OaVec<const char*> EnabledExtensions_;

	// Helper: Validate module dependencies
	OaStatus ValidateDependencies() const;

	// Helper: Sort modules by dependencies
	void SortModulesByDependencies();

	// Helper: Create logical device
	OaResult<VkDevice> CreateLogicalDevice(
		VkPhysicalDevice InPhysicalDevice,
		const OaVkQueuePlan& InQueuePlan
	);

	// Helper: Populate device info
	void PopulateDeviceInfo(
		VkPhysicalDevice InPhysicalDevice,
		VkDevice InDevice,
		OaVkDevice& OutDevice
	);
};
