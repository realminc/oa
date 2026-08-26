#include <oa/runtime/instance.h>

#include <oa/core/log.h>
#include <oa/core/std/cString.h>
#include <oa/core/std/string.h>

#include <stdlib.h>

namespace {

enum class ValidationMode {
	Core,
	Synchronization,
	GpuAssisted,
	All,
};

oa::Result<ValidationMode> requestedValidationMode() {
	const char* value = ::getenv("OA_VK_VALIDATION_MODE");
	if (value == nullptr || value[0] == '\0' || oa::strcmp(value, "sync") == 0) {
		return ValidationMode::Synchronization;
	}
	if (oa::strcmp(value, "core") == 0) {
		return ValidationMode::Core;
	}
	if (oa::strcmp(value, "gpu") == 0) {
		return ValidationMode::GpuAssisted;
	}
	if (oa::strcmp(value, "all") == 0) {
		return ValidationMode::All;
	}
	return oa::Status::error(
		oa::StatusCode::InvalidArgument,
		"OA_VK_VALIDATION_MODE must be one of: core, sync, gpu, all");
}

const char* validationModeName(ValidationMode inMode) {
	switch (inMode) {
		case ValidationMode::Core: return "core";
		case ValidationMode::Synchronization: return "core,synchronization";
		case ValidationMode::GpuAssisted: return "core,gpu-assisted";
		case ValidationMode::All: return "core,synchronization,gpu-assisted";
	}
	return "core";
}

} // namespace


oa::Result<VkInstance> oavk::Instance::createInstance(
	oa::StringView inAppName,
	oa::U32 inAppVersionPatch,
	oa::Bool inEnableValidation,
	oa::Span<const char* const> inExtraInstanceExtensions,
	oa::Bool inWantsPresentation
) {
	// Embedders such as the android mobile runtime may select a per-app vulkan
	// implementation first via oaVkInitCustom. Preserve that dispatch instead
	// of silently reopening android's system loader here.
	VkResult vkInitResult = VK_SUCCESS;
	if (oaVkGetInstanceProcAddr() == nullptr) {
		vkInitResult = oaVkInit();
	}
	if (vkInitResult != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,	"oaVkInit failed — no vulkan loader on system?");
	}

	oa::String appNameCopy(inAppName);
	const oa::U32 patchClamped = (inAppVersionPatch > 4095u) ? 4095u : inAppVersionPatch;

	VkApplicationInfo appInfo = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = appNameCopy.cStr(),
		.applicationVersion = VK_MAKE_VERSION(0, 0, patchClamped),
		.pEngineName = "OA",
		.engineVersion = VK_MAKE_VERSION(0, 2, 0),
		.apiVersion = oavk::MinApiVersion,
	};

	oa::Vec<const char*> extNames;
	auto appendUnique = [&extNames](const char* inName) {
		for (const char* existing : extNames) {
			if (oa::strcmp(existing, inName) == 0) return;
		}
		extNames.pushBack(inName);
	};
	for (oa::U32 bi = 0; bi < oavk::NumInstanceExtensions; ++bi) {
		appendUnique(oavk::InstanceExtensionNames.data()[bi]);
	}
	for (oa::U32 ei = 0; ei < inExtraInstanceExtensions.size(); ++ei) {
		const char* extra = inExtraInstanceExtensions.data()[ei];
		if (!extra || !extra[0]) {
			continue;
		}
		appendUnique(extra);
	}

	// Swapchain-maintenance present fences have an instance-side dependency.
	// Enable every available maintenance spelling so the later device probe can
	// select KHR or EXT without creating an invalid extension combination.
	if (inWantsPresentation) {
		oa::U32 count = 0;
		if (oaVkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr)
			== VK_SUCCESS && count > 0) {
			oa::Vec<VkExtensionProperties> properties(count);
			if (oaVkEnumerateInstanceExtensionProperties(
				nullptr, &count, properties.data()) == VK_SUCCESS) {
				bool hasSurfaceCaps2 = false;
				bool hasKhrMaintenance = false;
				bool hasExtMaintenance = false;
				for (const auto& property : properties) {
					hasSurfaceCaps2 |= oa::strcmp(property.extensionName,
						VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) == 0;
					hasKhrMaintenance |= oa::strcmp(property.extensionName,
						VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME) == 0;
					hasExtMaintenance |= oa::strcmp(property.extensionName,
						VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME) == 0;
				}
				if (hasSurfaceCaps2) {
					appendUnique(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
					if (hasKhrMaintenance) {
						appendUnique(VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
					}
					if (hasExtMaintenance) {
						appendUnique(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
					}
				}
			}
		}
	}

	ValidationMode validationMode = ValidationMode::Synchronization;
	bool enableValidationFeatures = false;
	if (inEnableValidation) {
		auto modeResult = requestedValidationMode();
		if (!modeResult.isOk()) {
			return modeResult.getStatus();
		}
		validationMode = modeResult.getValue();

		// VK_EXT_validation_features is exposed by the validation layer, not
		// necessarily as a global instance extension. query that layer directly.
		oa::U32 propertyCount = 0;
		if (oaVkEnumerateInstanceExtensionProperties(
				oavk::InstanceLayerNames[0], &propertyCount, nullptr) == VK_SUCCESS
			&& propertyCount > 0)
		{
			oa::Vec<VkExtensionProperties> properties(propertyCount);
			if (oaVkEnumerateInstanceExtensionProperties(
					oavk::InstanceLayerNames[0], &propertyCount, properties.data()) == VK_SUCCESS)
			{
				for (const auto& property : properties) {
					if (oa::strcmp(property.extensionName,
							VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0)
					{
						enableValidationFeatures = true;
						break;
					}
				}
			}
		}
		if (validationMode != ValidationMode::Core && !enableValidationFeatures) {
			return oa::Status::error(
				oa::StatusCode::FailedPrecondition,
				"requested vulkan validation features are unavailable");
		}
		if (enableValidationFeatures) {
			appendUnique(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		}
	}

	const oa::U32 layerCount = inEnableValidation ? static_cast<oa::U32>(oavk::InstanceLayerNames.size()) : 0u;
	const char* const* layerNames =	(layerCount > 0u) ? oavk::InstanceLayerNames.data() : nullptr;
	const oa::U32 instExtCount = extNames.size();
	const char* const* instExtNames =	(instExtCount > 0u) ? extNames.data() : nullptr;

	VkInstanceCreateInfo instCI{};
	instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instCI.pApplicationInfo = &appInfo;
	instCI.enabledLayerCount = layerCount;
	instCI.ppEnabledLayerNames = layerNames;
	instCI.enabledExtensionCount = instExtCount;
	instCI.ppEnabledExtensionNames = instExtNames;

	VkValidationFeatureEnableEXT enabledValidationFeatures[3]{};
	oa::U32 enabledValidationFeatureCount = 0;
	if (validationMode == ValidationMode::Synchronization
		or validationMode == ValidationMode::All)
	{
		enabledValidationFeatures[enabledValidationFeatureCount++] =
			VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
	}
	if (validationMode == ValidationMode::GpuAssisted
		or validationMode == ValidationMode::All)
	{
		enabledValidationFeatures[enabledValidationFeatureCount++] =
			VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT;
		enabledValidationFeatures[enabledValidationFeatureCount++] =
			VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT;
	}
	VkValidationFeaturesEXT validationFeatures{};
	if (enableValidationFeatures && enabledValidationFeatureCount > 0) {
		validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
		validationFeatures.enabledValidationFeatureCount = enabledValidationFeatureCount;
		validationFeatures.pEnabledValidationFeatures = enabledValidationFeatures;
		instCI.pNext = &validationFeatures;
	}

	VkInstance instance = VK_NULL_HANDLE;
	VkResult r = oaVkCreateInstance(&instCI, nullptr, &instance);
	if (r != VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,	oa::String("vkCreateInstance failed (VkResult=") + oa::toString(static_cast<oa::I64>(r)) + ")");
	}
	if (inEnableValidation) {
		OaLogInfo(oa::LogComponent::Runtime, "Validation features: %s",
			validationModeName(validationMode));
	}
	return oa::Result<VkInstance>(instance);
}

void oavk::Instance::destroyInstance(
	const OaVkInstanceTable& inDispatch,
	VkInstance inInstance) noexcept
{
	if (inInstance != VK_NULL_HANDLE) {
		inDispatch.vkDestroyInstance(inInstance, nullptr);
	}
}
