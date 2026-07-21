#include <Oa/Runtime/Instance.h>

#include <Oa/Core/Log.h>
#include <Oa/Core/Std/String.h>

#include <cstdlib>
#include <cstring>

namespace {

enum class OaValidationMode {
	Core,
	Synchronization,
	GpuAssisted,
	All,
};

OaResult<OaValidationMode> OaRequestedValidationMode() {
	const char* value = std::getenv("OA_VK_VALIDATION_MODE");
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "sync") == 0) {
		return OaValidationMode::Synchronization;
	}
	if (std::strcmp(value, "core") == 0) {
		return OaValidationMode::Core;
	}
	if (std::strcmp(value, "gpu") == 0) {
		return OaValidationMode::GpuAssisted;
	}
	if (std::strcmp(value, "all") == 0) {
		return OaValidationMode::All;
	}
	return OaStatus::Error(
		OaStatusCode::InvalidArgument,
		"OA_VK_VALIDATION_MODE must be one of: core, sync, gpu, all");
}

const char* OaValidationModeName(OaValidationMode InMode) {
	switch (InMode) {
		case OaValidationMode::Core: return "core";
		case OaValidationMode::Synchronization: return "core,synchronization";
		case OaValidationMode::GpuAssisted: return "core,gpu-assisted";
		case OaValidationMode::All: return "core,synchronization,gpu-assisted";
	}
	return "core";
}

} // namespace


OaResult<VkInstance> OaVkInstance::CreateInstance(
	OaStringView InAppName,
	OaU32 InAppVersionPatch,
	OaBool InEnableValidation,
	OaSpan<const char* const> InExtraInstanceExtensions,
	OaBool InWantsPresentation
) {
	// Embedders such as the Android mobile runtime may select a per-app Vulkan
	// implementation first via OaVkInitCustom. Preserve that dispatch instead
	// of silently reopening Android's system loader here.
	VkResult vkInitResult = VK_SUCCESS;
	if (vkGetInstanceProcAddr == nullptr) {
		vkInitResult = OaVkInit();
	}
	if (vkInitResult != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,	"OaVkInit failed — no Vulkan loader on system?");
	}

	OaString appNameCopy(InAppName);
	const OaU32 patchClamped = (InAppVersionPatch > 4095u) ? 4095u : InAppVersionPatch;

	VkApplicationInfo appInfo = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = appNameCopy.c_str(),
		.applicationVersion = VK_MAKE_VERSION(0, 0, patchClamped),
		.pEngineName = "OA",
		.engineVersion = VK_MAKE_VERSION(0, 2, 0),
		.apiVersion = OaVkMinApiVersion,
	};

	OaVec<const char*> extNames;
	auto appendUnique = [&extNames](const char* InName) {
		for (const char* existing : extNames) {
			if (std::strcmp(existing, InName) == 0) return;
		}
		extNames.PushBack(InName);
	};
	for (OaU32 bi = 0; bi < OaVkNumInstanceExtensions; ++bi) {
		appendUnique(OaVkInstanceExtensionNames[bi]);
	}
	for (OaU32 ei = 0; ei < InExtraInstanceExtensions.Size(); ++ei) {
		const char* extra = InExtraInstanceExtensions.Data()[ei];
		if (!extra || !extra[0]) {
			continue;
		}
		appendUnique(extra);
	}

	// Swapchain-maintenance present fences have an instance-side dependency.
	// Enable every available maintenance spelling so the later device probe can
	// select KHR or EXT without creating an invalid extension combination.
	if (InWantsPresentation) {
		OaU32 count = 0;
		if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr)
			== VK_SUCCESS && count > 0) {
			OaVec<VkExtensionProperties> properties(count);
			if (vkEnumerateInstanceExtensionProperties(
				nullptr, &count, properties.Data()) == VK_SUCCESS) {
				bool hasSurfaceCaps2 = false;
				bool hasKhrMaintenance = false;
				bool hasExtMaintenance = false;
				for (const auto& property : properties) {
					hasSurfaceCaps2 |= std::strcmp(property.extensionName,
						VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) == 0;
					hasKhrMaintenance |= std::strcmp(property.extensionName,
						VK_KHR_SURFACE_MAINTENANCE_1_EXTENSION_NAME) == 0;
					hasExtMaintenance |= std::strcmp(property.extensionName,
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

	OaValidationMode validationMode = OaValidationMode::Synchronization;
	bool enableValidationFeatures = false;
	if (InEnableValidation) {
		auto modeResult = OaRequestedValidationMode();
		if (!modeResult.IsOk()) {
			return modeResult.GetStatus();
		}
		validationMode = modeResult.GetValue();

		// VK_EXT_validation_features is exposed by the validation layer, not
		// necessarily as a global instance extension. Query that layer directly.
		OaU32 propertyCount = 0;
		if (vkEnumerateInstanceExtensionProperties(
				OaVkInstanceLayerNames[0], &propertyCount, nullptr) == VK_SUCCESS
			&& propertyCount > 0)
		{
			OaVec<VkExtensionProperties> properties(propertyCount);
			if (vkEnumerateInstanceExtensionProperties(
					OaVkInstanceLayerNames[0], &propertyCount, properties.Data()) == VK_SUCCESS)
			{
				for (const auto& property : properties) {
					if (std::strcmp(property.extensionName,
							VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0)
					{
						enableValidationFeatures = true;
						break;
					}
				}
			}
		}
		if (validationMode != OaValidationMode::Core && !enableValidationFeatures) {
			return OaStatus::Error(
				OaStatusCode::FailedPrecondition,
				"requested Vulkan validation features are unavailable");
		}
		if (enableValidationFeatures) {
			appendUnique(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
		}
	}

	const OaU32 layerCount = InEnableValidation ? static_cast<OaU32>(OaVkInstanceLayerNames.size()) : 0u;
	const char* const* layerNames =	(layerCount > 0u) ? OaVkInstanceLayerNames.data() : nullptr;
	const OaU32 instExtCount = extNames.Size();
	const char* const* instExtNames =	(instExtCount > 0u) ? extNames.Data() : nullptr;

	VkInstanceCreateInfo instCI{};
	instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instCI.pApplicationInfo = &appInfo;
	instCI.enabledLayerCount = layerCount;
	instCI.ppEnabledLayerNames = layerNames;
	instCI.enabledExtensionCount = instExtCount;
	instCI.ppEnabledExtensionNames = instExtNames;

	VkValidationFeatureEnableEXT enabledValidationFeatures[3]{};
	OaU32 enabledValidationFeatureCount = 0;
	if (validationMode == OaValidationMode::Synchronization
		or validationMode == OaValidationMode::All)
	{
		enabledValidationFeatures[enabledValidationFeatureCount++] =
			VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
	}
	if (validationMode == OaValidationMode::GpuAssisted
		or validationMode == OaValidationMode::All)
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
	VkResult r = vkCreateInstance(&instCI, nullptr, &instance);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,	OaString("vkCreateInstance failed (VkResult=") + OaToString(static_cast<OaI64>(r)) + ")");
	}
	if (InEnableValidation) {
		OA_LOG_INFO(OaLogComponent::Core, "Validation features: %s",
			OaValidationModeName(validationMode));
	}
	return OaResult<VkInstance>(instance);
}

void OaVkInstance::DestroyInstance(VkInstance InInstance) noexcept {
	if (InInstance != VK_NULL_HANDLE) {
		vkDestroyInstance(InInstance, nullptr);
	}
}
