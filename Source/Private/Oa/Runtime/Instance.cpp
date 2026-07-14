#include <Oa/Runtime/Instance.h>

#include <Oa/Core/Std/String.h>

#include <cstring>


OaResult<VkInstance> OaVkInstance::CreateInstance(
	OaStringView InAppName,
	OaU32 InAppVersionPatch,
	OaBool InEnableValidation,
	OaSpan<const char* const> InExtraInstanceExtensions
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
	for (OaU32 bi = 0; bi < OaVkNumInstanceExtensions; ++bi) {
		extNames.PushBack(OaVkInstanceExtensionNames[bi]);
	}
	for (OaU32 ei = 0; ei < InExtraInstanceExtensions.Size(); ++ei) {
		const char* extra = InExtraInstanceExtensions.Data()[ei];
		if (!extra || !extra[0]) {
			continue;
		}
		bool duplicate = false;
		for (OaU32 j = 0; j < extNames.Size(); ++j) {
			if (std::strcmp(extNames[j], extra) == 0) {
				duplicate = true;
				break;
			}
		}
		if (!duplicate) {
			extNames.PushBack(extra);
		}
	}

	bool enableSyncValidation = false;
	if (InEnableValidation) {
		OaU32 propertyCount = 0;
		if (vkEnumerateInstanceExtensionProperties(
				nullptr, &propertyCount, nullptr) == VK_SUCCESS
			&& propertyCount > 0)
		{
			OaVec<VkExtensionProperties> properties(propertyCount);
			if (vkEnumerateInstanceExtensionProperties(
					nullptr, &propertyCount, properties.Data()) == VK_SUCCESS)
			{
				for (const auto& property : properties) {
					if (std::strcmp(property.extensionName,
							VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0)
					{
						enableSyncValidation = true;
						break;
					}
				}
			}
		}
		if (enableSyncValidation) {
			bool duplicate = false;
			for (const char* extension : extNames) {
				if (std::strcmp(extension,
						VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME) == 0)
				{
					duplicate = true;
					break;
				}
			}
			if (!duplicate) {
				extNames.PushBack(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
			}
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

	VkValidationFeatureEnableEXT syncValidation =
		VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
	VkValidationFeaturesEXT validationFeatures{};
	if (enableSyncValidation) {
		validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
		validationFeatures.enabledValidationFeatureCount = 1;
		validationFeatures.pEnabledValidationFeatures = &syncValidation;
		instCI.pNext = &validationFeatures;
	}

	VkInstance instance = VK_NULL_HANDLE;
	VkResult r = vkCreateInstance(&instCI, nullptr, &instance);
	if (r != VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError,	OaString("vkCreateInstance failed (VkResult=") + OaToString(static_cast<OaI64>(r)) + ")");
	}
	return OaResult<VkInstance>(instance);
}

void OaVkInstance::DestroyInstance(VkInstance InInstance) noexcept {
	if (InInstance != VK_NULL_HANDLE) {
		vkDestroyInstance(InInstance, nullptr);
	}
}
