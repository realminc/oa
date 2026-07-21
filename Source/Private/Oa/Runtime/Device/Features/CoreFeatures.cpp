// OA Vulkan Core Features Module
// Handles core Vulkan 1.3 features required by all OA devices:
// - Bindless (descriptor indexing)
// - Buffer device address
// - Timeline semaphores
// - Synchronization2

#include "../FeatureModule.h"
#include <Oa/Core/Log.h>
#include <cstring>


class OaVkCoreFeatures : public OaVkFeatureModule {
public:
	OaStringView Name() const override {
		return "Core";
	}

	void ProbeExtensions(
		const OaVec<VkExtensionProperties>& InAvailableExtensions,
		OaVkPhysExtProbe& OutProbe
	) override {
		// Core features don't require additional extensions beyond Vulkan 1.3
		// Pipeline library is optional but recommended
		for (const auto& ext : InAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
				OutProbe.KhrBufferDeviceAddress = true;
			}
			else if (strcmp(ext.extensionName, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) == 0) {
				OutProbe.ExtDescriptorIndexing = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
				OutProbe.KhrTimelineSemaphore = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
				OutProbe.KhrSynchronization2 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0) {
				OutProbe.KhrDynamicRendering = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) == 0) {
				OutProbe.PipelineLibrary = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) == 0) {
				OutProbe.ExternalMemory = true;
			}
#ifdef VK_KHR_external_memory_fd
			else if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) {
				OutProbe.ExternalMemoryFd = true;
			}
#endif
#ifdef VK_EXT_external_memory_dma_buf
			else if (strcmp(ext.extensionName, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) == 0) {
				OutProbe.ExternalMemoryDmaBuf = true;
			}
#endif
#ifdef VK_EXT_image_drm_format_modifier
			else if (strcmp(ext.extensionName, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) == 0) {
				OutProbe.ImageDrmFormatModifier = true;
			}
#endif
#ifdef VK_EXT_queue_family_foreign
			else if (strcmp(ext.extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0) {
				OutProbe.QueueFamilyForeign = true;
			}
#endif
		}
	}

	void QueryFeatures(
		VkPhysicalDevice InPhysicalDevice,
		OaVkDeviceFeatureBundle& OutBundle
	) override {
		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(InPhysicalDevice, &properties);
		OutBundle.PhysicalApiVersion = properties.apiVersion;

		// Set up feature query chain
		OutBundle.SupportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		OutBundle.Supported11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		OutBundle.Supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		OutBundle.Supported13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		OutBundle.SupportedFeatures2.pNext = &OutBundle.Supported11;
		OutBundle.Supported11.pNext = &OutBundle.Supported12;
		OutBundle.Supported12.pNext = &OutBundle.Supported13;
		OutBundle.Supported13.pNext = nullptr;

		vkGetPhysicalDeviceFeatures2(InPhysicalDevice, &OutBundle.SupportedFeatures2);

		// Android Vulkan HAL drivers may advertise a 1.0/1.1 physical-device API
		// while providing the promoted ModernCompute contract through KHR/EXT
		// feature structs. Query that extension path independently: Vulkan forbids
		// mixing promoted aggregate and individual structs in one pNext chain.
		VkPhysicalDeviceFeatures2 extensionFeatures2{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		};
		OutBundle.SupportedBufferDeviceAddress = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
		};
		OutBundle.SupportedDescriptorIndexing = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
		};
		OutBundle.SupportedTimelineSemaphore = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
		};
		OutBundle.SupportedSynchronization2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
		};
		OutBundle.SupportedDynamicRendering = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
		};
		extensionFeatures2.pNext = &OutBundle.SupportedBufferDeviceAddress;
		OutBundle.SupportedBufferDeviceAddress.pNext = &OutBundle.SupportedDescriptorIndexing;
		OutBundle.SupportedDescriptorIndexing.pNext = &OutBundle.SupportedTimelineSemaphore;
		OutBundle.SupportedTimelineSemaphore.pNext = &OutBundle.SupportedSynchronization2;
		OutBundle.SupportedSynchronization2.pNext = &OutBundle.SupportedDynamicRendering;
		vkGetPhysicalDeviceFeatures2(InPhysicalDevice, &extensionFeatures2);

		if (VK_API_VERSION_MAJOR(OutBundle.PhysicalApiVersion) < 1 ||
			(VK_API_VERSION_MAJOR(OutBundle.PhysicalApiVersion) == 1 &&
			 VK_API_VERSION_MINOR(OutBundle.PhysicalApiVersion) < 2)) {
			OutBundle.Supported12.bufferDeviceAddress =
				OutBundle.SupportedBufferDeviceAddress.bufferDeviceAddress;
			OutBundle.Supported12.descriptorIndexing =
				OutBundle.SupportedDescriptorIndexing.runtimeDescriptorArray;
			OutBundle.Supported12.runtimeDescriptorArray =
				OutBundle.SupportedDescriptorIndexing.runtimeDescriptorArray;
			OutBundle.Supported12.descriptorBindingPartiallyBound =
				OutBundle.SupportedDescriptorIndexing.descriptorBindingPartiallyBound;
			OutBundle.Supported12.descriptorBindingVariableDescriptorCount =
				OutBundle.SupportedDescriptorIndexing.descriptorBindingVariableDescriptorCount;
			OutBundle.Supported12.shaderSampledImageArrayNonUniformIndexing =
				OutBundle.SupportedDescriptorIndexing.shaderSampledImageArrayNonUniformIndexing;
			OutBundle.Supported12.shaderStorageBufferArrayNonUniformIndexing =
				OutBundle.SupportedDescriptorIndexing.shaderStorageBufferArrayNonUniformIndexing;
			OutBundle.Supported12.descriptorBindingStorageBufferUpdateAfterBind =
				OutBundle.SupportedDescriptorIndexing.descriptorBindingStorageBufferUpdateAfterBind;
			OutBundle.Supported12.descriptorBindingStorageImageUpdateAfterBind =
				OutBundle.SupportedDescriptorIndexing.descriptorBindingStorageImageUpdateAfterBind;
			OutBundle.Supported12.descriptorBindingSampledImageUpdateAfterBind =
				OutBundle.SupportedDescriptorIndexing.descriptorBindingSampledImageUpdateAfterBind;
			OutBundle.Supported12.descriptorBindingUpdateUnusedWhilePending =
				OutBundle.SupportedDescriptorIndexing.descriptorBindingUpdateUnusedWhilePending;
			OutBundle.Supported12.timelineSemaphore =
				OutBundle.SupportedTimelineSemaphore.timelineSemaphore;
		}
		if (VK_API_VERSION_MAJOR(OutBundle.PhysicalApiVersion) < 1 ||
			(VK_API_VERSION_MAJOR(OutBundle.PhysicalApiVersion) == 1 &&
			 VK_API_VERSION_MINOR(OutBundle.PhysicalApiVersion) < 3)) {
			OutBundle.Supported13.synchronization2 =
				OutBundle.SupportedSynchronization2.synchronization2;
			OutBundle.Supported13.dynamicRendering =
				OutBundle.SupportedDynamicRendering.dynamicRendering;
		}

		// Validate core requirements
		if (!OutBundle.Supported12.bufferDeviceAddress) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"Device does not support bufferDeviceAddress (required)");
		}
		if (!OutBundle.Supported12.descriptorIndexing) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"Device does not support descriptorIndexing (required)");
		}
		if (!OutBundle.Supported12.timelineSemaphore) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"Device does not support timelineSemaphore (required)");
		}
		if (!OutBundle.Supported13.synchronization2) {
			OA_LOG_ERROR(OaLogComponent::Core,
				"Device does not support synchronization2 (required)");
		}

		// Check for 16-bit storage (optional but recommended)
		OutBundle.Has16bit = OutBundle.Supported11.storageBuffer16BitAccess == VK_TRUE;
	}

	void BuildFeatureChain(
		OaVkDeviceFeatureBundle& InOutBundle
	) override {
		const OaU32 apiMajor = VK_API_VERSION_MAJOR(InOutBundle.PhysicalApiVersion);
		const OaU32 apiMinor = VK_API_VERSION_MINOR(InOutBundle.PhysicalApiVersion);
		const bool core11 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 1);
		const bool core12 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 2);
		const bool core13 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 3);

		// Enable core features, or their binary-equivalent extension providers on
		// older Android HAL API versions.
		InOutBundle.Features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		InOutBundle.Features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		InOutBundle.Features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		InOutBundle.Features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		VkBaseOutStructure* tail = reinterpret_cast<VkBaseOutStructure*>(&InOutBundle.Features2);
		tail->pNext = nullptr;
		auto append = [&tail](auto& feature) {
			auto* node = reinterpret_cast<VkBaseOutStructure*>(&feature);
			node->pNext = nullptr;
			tail->pNext = node;
			tail = node;
		};
		if (core11) append(InOutBundle.Features11);
		if (core12) append(InOutBundle.Features12);
		if (core13) append(InOutBundle.Features13);

		// Enable the storage capabilities emitted by the common shader compiler.
		// shaderInt8/shaderInt16 control arithmetic; the storage feature bits are
		// separate requirements for SPIR-V UniformAndStorageBuffer{8,16}BitAccess.
		if (InOutBundle.Supported11.storageBuffer16BitAccess) {
			InOutBundle.Features11.storageBuffer16BitAccess = VK_TRUE;
		}
		if (InOutBundle.Supported11.uniformAndStorageBuffer16BitAccess) {
			InOutBundle.Features11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
		}
		if (InOutBundle.Supported11.storagePushConstant16) {
			InOutBundle.Features11.storagePushConstant16 = VK_TRUE;
		}
		// Graphics shaders emitted by the shared Slang pipeline declare the
		// DrawParameters capability (base vertex/instance and draw index). The
		// feature is core in Vulkan 1.1 but still has to be enabled explicitly.
		if (InOutBundle.Supported11.shaderDrawParameters) {
			InOutBundle.Features11.shaderDrawParameters = VK_TRUE;
		}

		// Enable required Vulkan 1.2 features
		InOutBundle.Features12.bufferDeviceAddress = VK_TRUE;
		InOutBundle.Features12.descriptorIndexing = VK_TRUE;
		InOutBundle.Features12.runtimeDescriptorArray = VK_TRUE;
		InOutBundle.Features12.descriptorBindingPartiallyBound = VK_TRUE;
		InOutBundle.Features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
		InOutBundle.Features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		InOutBundle.Features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
		// The bindless heap creates its descriptor set layouts with
		// UPDATE_AFTER_BIND. That flag requires the matching descriptorType
		// UpdateAfterBind feature to be enabled at device-create time
		// (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBinding…-03006/07/08).
		InOutBundle.Features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
		InOutBundle.Features12.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
		InOutBundle.Features12.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
		InOutBundle.Features12.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;
		InOutBundle.Features12.timelineSemaphore = VK_TRUE;

		// Enable optional Vulkan 1.2 features if available
		if (InOutBundle.Supported12.shaderFloat16) {
			InOutBundle.Features12.shaderFloat16 = VK_TRUE;
		}
		if (InOutBundle.Supported12.shaderInt8) {
			InOutBundle.Features12.shaderInt8 = VK_TRUE;
		}
		if (InOutBundle.Supported12.storageBuffer8BitAccess) {
			InOutBundle.Features12.storageBuffer8BitAccess = VK_TRUE;
		}
		if (InOutBundle.Supported12.uniformAndStorageBuffer8BitAccess) {
			InOutBundle.Features12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
		}
		if (InOutBundle.Supported12.storagePushConstant8) {
			InOutBundle.Features12.storagePushConstant8 = VK_TRUE;
		}
		if (InOutBundle.Supported12.vulkanMemoryModel) {
			InOutBundle.Features12.vulkanMemoryModel = VK_TRUE;
		}
		if (InOutBundle.Supported12.vulkanMemoryModel
			&& InOutBundle.Supported12.vulkanMemoryModelDeviceScope)
		{
			InOutBundle.Features12.vulkanMemoryModelDeviceScope = VK_TRUE;
		}

		// Enable required Vulkan 1.3 features
		InOutBundle.Features13.synchronization2 = VK_TRUE;
		InOutBundle.Features13.dynamicRendering = VK_TRUE;

		if (!core12) {
			InOutBundle.BufferDeviceAddressFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
				.bufferDeviceAddress = VK_TRUE,
			};
			InOutBundle.DescriptorIndexingFeatures = InOutBundle.SupportedDescriptorIndexing;
			InOutBundle.DescriptorIndexingFeatures.pNext = nullptr;
			InOutBundle.TimelineSemaphoreFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
				.timelineSemaphore = VK_TRUE,
			};
			append(InOutBundle.BufferDeviceAddressFeatures);
			append(InOutBundle.DescriptorIndexingFeatures);
			append(InOutBundle.TimelineSemaphoreFeatures);
		}
		if (!core13) {
			InOutBundle.Synchronization2Features = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
				.synchronization2 = VK_TRUE,
			};
			InOutBundle.DynamicRenderingFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
				.dynamicRendering = VK_TRUE,
			};
			append(InOutBundle.Synchronization2Features);
			append(InOutBundle.DynamicRenderingFeatures);
		}

		// Enable optional Vulkan 1.3 features if available
		if (InOutBundle.Supported13.maintenance4) {
			InOutBundle.Features13.maintenance4 = VK_TRUE;
		}

		// subgroupSizeControl lets a compute pipeline pin its subgroup (SIMD)
		// width via VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Used by
		// the OA_GEMM_SUBGROUP_SIZE experiment to align to the Intel Xe native
		// fp32 vector width (SIMD16). Harmless capability when the knob is unset.
		if (InOutBundle.Supported13.subgroupSizeControl) {
			InOutBundle.Features13.subgroupSizeControl = VK_TRUE;
		}

		// Enable base features
		if (InOutBundle.SupportedFeatures2.features.shaderInt64) {
			InOutBundle.Features2.features.shaderInt64 = VK_TRUE;
		}
		if (InOutBundle.SupportedFeatures2.features.shaderInt16) {
			InOutBundle.Features2.features.shaderInt16 = VK_TRUE;
		}
	}

	void CollectExtensions(
		const OaVkPhysExtProbe& InProbe,
		const OaVkDeviceFeatureBundle& InBundle,
		OaVec<const char*>& OutExtensions
	) override {
		const OaU32 apiMajor = VK_API_VERSION_MAJOR(InBundle.PhysicalApiVersion);
		const OaU32 apiMinor = VK_API_VERSION_MINOR(InBundle.PhysicalApiVersion);
		const bool core12 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 2);
		const bool core13 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 3);
		if (!core12) {
			if (InProbe.KhrBufferDeviceAddress) {
				OutExtensions.PushBack(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
			}
			if (InProbe.ExtDescriptorIndexing) {
				OutExtensions.PushBack(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
			}
			if (InProbe.KhrTimelineSemaphore) {
				OutExtensions.PushBack(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
			}
		}
		if (!core13) {
			if (InProbe.KhrSynchronization2) {
				OutExtensions.PushBack(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
			}
			if (InProbe.KhrDynamicRendering) {
				OutExtensions.PushBack(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
			}
		}

		// Optional extensions
		if (InProbe.PipelineLibrary) {
			OutExtensions.PushBack(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
		}
		if (InProbe.ExternalMemory && InProbe.ExternalMemoryFd) {
			OutExtensions.PushBack(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef VK_KHR_external_memory_fd
			OutExtensions.PushBack(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
		}
		if (InProbe.ExternalMemoryDmaBuf) {
			OutExtensions.PushBack(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
		}
		if (InProbe.ImageDrmFormatModifier) {
			OutExtensions.PushBack(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);
		}
		if (InProbe.QueueFamilyForeign) {
			OutExtensions.PushBack(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
		}
	}

	OaVec<OaStringView> Dependencies() const override {
		return {};  // Core has no dependencies
	}
};


// Factory function
OaUniquePtr<OaVkFeatureModule> OaVkCreateCoreFeatures() {
	return OaMakeUniquePtr<OaVkCoreFeatures>();
}
