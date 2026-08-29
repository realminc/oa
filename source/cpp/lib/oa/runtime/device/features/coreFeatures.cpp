// OA vulkan Core features Module
// Handles core vulkan 1.3 features required by all OA devices:
// - bindless (descriptor indexing)
// - Buffer device address
// - Timeline semaphores
// - synchronization2

#include "../featureModule.h"
#include <oa/core/log.h>
#include <string.h>


class CoreFeatures : public oavk::FeatureModule {
public:
	oa::StringView name() const override {
		return "Core";
	}

	void probeExtensions(
		const oa::Vector<VkExtensionProperties>& inAvailableExtensions,
		oavk::PhysicalExtensionProbe& outProbe
	) override {
		// Core features don't require additional extensions beyond vulkan 1.3
		// pipeline library is optional but recommended
		for (const auto& ext : inAvailableExtensions) {
			if (strcmp(ext.extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
				outProbe.khrBufferDeviceAddress = true;
			}
			else if (strcmp(ext.extensionName, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) == 0) {
				outProbe.extDescriptorIndexing = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME) == 0) {
				outProbe.khrTimelineSemaphore = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) == 0) {
				outProbe.khrSynchronization2 = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) == 0) {
				outProbe.khrDynamicRendering = true;
			}
			else if (strcmp(ext.extensionName, oavk::ExtKhrCalibratedTimestamps) == 0) {
				outProbe.khrCalibratedTimestamps = true;
			}
			else if (strcmp(ext.extensionName, oavk::ExtExtCalibratedTimestamps) == 0) {
				outProbe.extCalibratedTimestamps = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) == 0) {
				outProbe.pipelineLibrary = true;
			}
			else if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) == 0) {
				outProbe.externalMemory = true;
			}
#ifdef VK_KHR_external_memory_fd
			else if (strcmp(ext.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) {
				outProbe.externalMemoryFd = true;
			}
#endif
#ifdef VK_EXT_external_memory_dma_buf
			else if (strcmp(ext.extensionName, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) == 0) {
				outProbe.externalMemoryDmaBuf = true;
			}
#endif
#ifdef VK_EXT_image_drm_format_modifier
			else if (strcmp(ext.extensionName, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME) == 0) {
				outProbe.imageDrmFormatModifier = true;
			}
#endif
#ifdef VK_EXT_queue_family_foreign
			else if (strcmp(ext.extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0) {
				outProbe.queueFamilyForeign = true;
			}
#endif
		}
	}

	void queryFeatures(
		const VklInstanceTable& inDispatch,
		VkPhysicalDevice inPhysicalDevice,
		oavk::DeviceFeatureBundle& outBundle
	) override {
		VkPhysicalDeviceProperties properties{};
		inDispatch.vkGetPhysicalDeviceProperties(inPhysicalDevice, &properties);
		outBundle.physicalApiVersion = properties.apiVersion;

		// set up feature query chain
		outBundle.supportedFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		outBundle.supported11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		outBundle.supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		outBundle.supported13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		outBundle.supportedFeatures2.pNext = &outBundle.supported11;
		outBundle.supported11.pNext = &outBundle.supported12;
		outBundle.supported12.pNext = &outBundle.supported13;
		outBundle.supported13.pNext = nullptr;

		inDispatch.vkGetPhysicalDeviceFeatures2(
			inPhysicalDevice, &outBundle.supportedFeatures2);

		// android vulkan HAL drivers may advertise a 1.0/1.1 physical-device API
		// while providing the promoted ModernCompute contract through KHR/EXT
		// feature structs. query that extension path independently: vulkan forbids
		// mixing promoted aggregate and individual structs in one pNext chain.
		VkPhysicalDeviceFeatures2 extensionFeatures2{
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
		};
		outBundle.supportedBufferDeviceAddress = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
		};
		outBundle.supportedDescriptorIndexing = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
		};
		outBundle.supportedTimelineSemaphore = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
		};
		outBundle.supportedSynchronization2 = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
		};
		outBundle.supportedDynamicRendering = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
		};
		outBundle.supportedSamplerYcbcrConversion = {
			.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
		};
		extensionFeatures2.pNext = &outBundle.supportedBufferDeviceAddress;
		outBundle.supportedBufferDeviceAddress.pNext = &outBundle.supportedDescriptorIndexing;
		outBundle.supportedDescriptorIndexing.pNext = &outBundle.supportedTimelineSemaphore;
		outBundle.supportedTimelineSemaphore.pNext = &outBundle.supportedSynchronization2;
		outBundle.supportedSynchronization2.pNext = &outBundle.supportedDynamicRendering;
		outBundle.supportedDynamicRendering.pNext = &outBundle.supportedSamplerYcbcrConversion;
		inDispatch.vkGetPhysicalDeviceFeatures2(
			inPhysicalDevice, &extensionFeatures2);

		if (VK_API_VERSION_MAJOR(outBundle.physicalApiVersion) < 1 ||
			(VK_API_VERSION_MAJOR(outBundle.physicalApiVersion) == 1 &&
			 VK_API_VERSION_MINOR(outBundle.physicalApiVersion) < 2)) {
			outBundle.supported12.bufferDeviceAddress =
				outBundle.supportedBufferDeviceAddress.bufferDeviceAddress;
			outBundle.supported12.descriptorIndexing =
				outBundle.supportedDescriptorIndexing.runtimeDescriptorArray;
			outBundle.supported12.runtimeDescriptorArray =
				outBundle.supportedDescriptorIndexing.runtimeDescriptorArray;
			outBundle.supported12.descriptorBindingPartiallyBound =
				outBundle.supportedDescriptorIndexing.descriptorBindingPartiallyBound;
			outBundle.supported12.descriptorBindingVariableDescriptorCount =
				outBundle.supportedDescriptorIndexing.descriptorBindingVariableDescriptorCount;
			outBundle.supported12.shaderSampledImageArrayNonUniformIndexing =
				outBundle.supportedDescriptorIndexing.shaderSampledImageArrayNonUniformIndexing;
			outBundle.supported12.shaderStorageBufferArrayNonUniformIndexing =
				outBundle.supportedDescriptorIndexing.shaderStorageBufferArrayNonUniformIndexing;
			outBundle.supported12.descriptorBindingStorageBufferUpdateAfterBind =
				outBundle.supportedDescriptorIndexing.descriptorBindingStorageBufferUpdateAfterBind;
			outBundle.supported12.descriptorBindingStorageImageUpdateAfterBind =
				outBundle.supportedDescriptorIndexing.descriptorBindingStorageImageUpdateAfterBind;
			outBundle.supported12.descriptorBindingSampledImageUpdateAfterBind =
				outBundle.supportedDescriptorIndexing.descriptorBindingSampledImageUpdateAfterBind;
			outBundle.supported12.descriptorBindingUpdateUnusedWhilePending =
				outBundle.supportedDescriptorIndexing.descriptorBindingUpdateUnusedWhilePending;
			outBundle.supported12.timelineSemaphore =
				outBundle.supportedTimelineSemaphore.timelineSemaphore;
		}
		if (VK_API_VERSION_MAJOR(outBundle.physicalApiVersion) < 1 ||
			(VK_API_VERSION_MAJOR(outBundle.physicalApiVersion) == 1 &&
			 VK_API_VERSION_MINOR(outBundle.physicalApiVersion) < 3)) {
			outBundle.supported13.synchronization2 =
				outBundle.supportedSynchronization2.synchronization2;
			outBundle.supported13.dynamicRendering =
				outBundle.supportedDynamicRendering.dynamicRendering;
		}

		// validate core requirements
		if (!outBundle.supported12.bufferDeviceAddress) {
			OaLogError(oa::LogComponent::Runtime,
				"Device does not support bufferDeviceAddress (required)");
		}
		if (!outBundle.supported12.descriptorIndexing) {
			OaLogError(oa::LogComponent::Runtime,
				"Device does not support descriptorIndexing (required)");
		}
		if (!outBundle.supported12.timelineSemaphore) {
			OaLogError(oa::LogComponent::Runtime,
				"Device does not support timelineSemaphore (required)");
		}
		if (!outBundle.supported13.synchronization2) {
			OaLogError(oa::LogComponent::Runtime,
				"Device does not support synchronization2 (required)");
		}

		// Check for 16-bit storage (optional but recommended)
		outBundle.has16bit = outBundle.supported11.storageBuffer16BitAccess == VK_TRUE;
	}

	void buildFeatureChain(
		oavk::DeviceFeatureBundle& inOutBundle
	) override {
		const oa::U32 apiMajor = VK_API_VERSION_MAJOR(inOutBundle.physicalApiVersion);
		const oa::U32 apiMinor = VK_API_VERSION_MINOR(inOutBundle.physicalApiVersion);
		const bool core11 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 1);
		const bool core12 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 2);
		const bool core13 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 3);

		// Enable core features, or their binary-equivalent extension providers on
		// older android HAL API versions.
		inOutBundle.features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
		inOutBundle.features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
		inOutBundle.features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
		inOutBundle.features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

		VkBaseOutStructure* tail = reinterpret_cast<VkBaseOutStructure*>(&inOutBundle.features2);
		tail->pNext = nullptr;
		auto append = [&tail](auto& feature) {
			auto* node = reinterpret_cast<VkBaseOutStructure*>(&feature);
			node->pNext = nullptr;
			tail->pNext = node;
			tail = node;
		};
		if (core11) append(inOutBundle.features11);
		if (core12) append(inOutBundle.features12);
		if (core13) append(inOutBundle.features13);

		// Enable the storage capabilities emitted by the common shader compiler.
		// shaderInt8/shaderInt16 control arithmetic; the storage feature bits are
		// separate requirements for SPIR-V UniformAndStorageBuffer{8,16}BitAccess.
		if (inOutBundle.supported11.storageBuffer16BitAccess) {
			inOutBundle.features11.storageBuffer16BitAccess = VK_TRUE;
		}
		if (inOutBundle.supported11.uniformAndStorageBuffer16BitAccess) {
			inOutBundle.features11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
		}
		if (inOutBundle.supported11.storagePushConstant16) {
			inOutBundle.features11.storagePushConstant16 = VK_TRUE;
		}
		// Graphics shaders emitted by the shared slang pipeline declare the
		// DrawParameters capability (base vertex/instance and draw index). The
		// feature is core in vulkan 1.1 but still has to be enabled explicitly.
		if (inOutBundle.supported11.shaderDrawParameters) {
			inOutBundle.features11.shaderDrawParameters = VK_TRUE;
		}
		if (core11 && inOutBundle.supported11.samplerYcbcrConversion) {
			inOutBundle.features11.samplerYcbcrConversion = VK_TRUE;
		}

		// Enable required vulkan 1.2 features
		inOutBundle.features12.bufferDeviceAddress = VK_TRUE;
		inOutBundle.features12.descriptorIndexing = VK_TRUE;
		inOutBundle.features12.runtimeDescriptorArray = VK_TRUE;
		inOutBundle.features12.descriptorBindingPartiallyBound = VK_TRUE;
		inOutBundle.features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
		inOutBundle.features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
		inOutBundle.features12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
		// The bindless heap creates its descriptor set layouts with
		// UPDATE_AFTER_BIND. That flag requires the matching descriptorType
		// UpdateAfterBind feature to be enabled at device-create time
		// (VUID-VkDescriptorSetLayoutBindingFlagsCreateInfo-descriptorBinding…-03006/07/08).
		inOutBundle.features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
		inOutBundle.features12.descriptorBindingStorageImageUpdateAfterBind  = VK_TRUE;
		inOutBundle.features12.descriptorBindingSampledImageUpdateAfterBind  = VK_TRUE;
		inOutBundle.features12.descriptorBindingUpdateUnusedWhilePending     = VK_TRUE;
		inOutBundle.features12.timelineSemaphore = VK_TRUE;

		// Enable optional vulkan 1.2 features if available
		if (inOutBundle.supported12.shaderFloat16) {
			inOutBundle.features12.shaderFloat16 = VK_TRUE;
		}
		if (inOutBundle.supported12.shaderInt8) {
			inOutBundle.features12.shaderInt8 = VK_TRUE;
		}
		if (inOutBundle.supported12.storageBuffer8BitAccess) {
			inOutBundle.features12.storageBuffer8BitAccess = VK_TRUE;
		}
		if (inOutBundle.supported12.uniformAndStorageBuffer8BitAccess) {
			inOutBundle.features12.uniformAndStorageBuffer8BitAccess = VK_TRUE;
		}
		if (inOutBundle.supported12.storagePushConstant8) {
			inOutBundle.features12.storagePushConstant8 = VK_TRUE;
		}
		if (inOutBundle.supported12.vulkanMemoryModel) {
			inOutBundle.features12.vulkanMemoryModel = VK_TRUE;
		}
		if (inOutBundle.supported12.vulkanMemoryModel
			&& inOutBundle.supported12.vulkanMemoryModelDeviceScope)
		{
			inOutBundle.features12.vulkanMemoryModelDeviceScope = VK_TRUE;
		}

		// Enable required vulkan 1.3 features
		inOutBundle.features13.synchronization2 = VK_TRUE;
		inOutBundle.features13.dynamicRendering = VK_TRUE;

		if (!core12) {
			inOutBundle.bufferDeviceAddressFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
				.bufferDeviceAddress = VK_TRUE,
			};
			inOutBundle.descriptorIndexingFeatures = inOutBundle.supportedDescriptorIndexing;
			inOutBundle.descriptorIndexingFeatures.pNext = nullptr;
			inOutBundle.timelineSemaphoreFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
				.timelineSemaphore = VK_TRUE,
			};
			append(inOutBundle.bufferDeviceAddressFeatures);
			append(inOutBundle.descriptorIndexingFeatures);
			append(inOutBundle.timelineSemaphoreFeatures);
		}
		if (!core13) {
			inOutBundle.synchronization2Features = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
				.synchronization2 = VK_TRUE,
			};
			inOutBundle.dynamicRenderingFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
				.dynamicRendering = VK_TRUE,
			};
			append(inOutBundle.synchronization2Features);
			append(inOutBundle.dynamicRenderingFeatures);
		}
		if (!core11 && inOutBundle.supportedSamplerYcbcrConversion.samplerYcbcrConversion) {
			inOutBundle.samplerYcbcrConversionFeatures = {
				.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES,
				.samplerYcbcrConversion = VK_TRUE,
			};
			append(inOutBundle.samplerYcbcrConversionFeatures);
		}

		// Enable optional vulkan 1.3 features if available
		if (inOutBundle.supported13.maintenance4) {
			inOutBundle.features13.maintenance4 = VK_TRUE;
		}

		// subgroupSizeControl lets a compute pipeline pin its subgroup (SIMD)
		// width via VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Used by
		// the OA_GEMM_SUBGROUP_SIZE experiment to align to the Intel Xe native
		// fp32 vector width (SIMD16). Harmless capability when the knob is unset.
		if (inOutBundle.supported13.subgroupSizeControl) {
			inOutBundle.features13.subgroupSizeControl = VK_TRUE;
		}

		// Enable base features
		if (inOutBundle.supportedFeatures2.features.shaderInt64) {
			inOutBundle.features2.features.shaderInt64 = VK_TRUE;
		}
		if (inOutBundle.supportedFeatures2.features.shaderInt16) {
			inOutBundle.features2.features.shaderInt16 = VK_TRUE;
		}
		if (inOutBundle.supportedFeatures2.features.shaderFloat64) {
			inOutBundle.features2.features.shaderFloat64 = VK_TRUE;
		}
	}

	void collectExtensions(
		const oavk::PhysicalExtensionProbe& inProbe,
		const oavk::DeviceFeatureBundle& inBundle,
		oa::Vector<const char*>& outExtensions
	) override {
		const oa::U32 apiMajor = VK_API_VERSION_MAJOR(inBundle.physicalApiVersion);
		const oa::U32 apiMinor = VK_API_VERSION_MINOR(inBundle.physicalApiVersion);
		const bool core12 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 2);
		const bool core13 = apiMajor > 1 || (apiMajor == 1 && apiMinor >= 3);
		if (!core12) {
			if (inProbe.khrBufferDeviceAddress) {
				outExtensions.pushBack(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
			}
			if (inProbe.extDescriptorIndexing) {
				outExtensions.pushBack(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
			}
			if (inProbe.khrTimelineSemaphore) {
				outExtensions.pushBack(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
			}
		}
		if (!core13) {
			if (inProbe.khrSynchronization2) {
				outExtensions.pushBack(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
			}
			if (inProbe.khrDynamicRendering) {
				outExtensions.pushBack(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
			}
		}

		// Optional extensions
		if (inProbe.pipelineLibrary) {
			outExtensions.pushBack(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
		}
		if (inProbe.externalMemory && inProbe.externalMemoryFd) {
			outExtensions.pushBack(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#ifdef VK_KHR_external_memory_fd
			outExtensions.pushBack(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
		}
		if (inProbe.externalMemoryDmaBuf) {
			outExtensions.pushBack(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
		}
		if (inProbe.imageDrmFormatModifier) {
			outExtensions.pushBack(VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);
		}
		if (inProbe.queueFamilyForeign) {
			outExtensions.pushBack(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
		}
		// KHR and EXT expose the same ABI. Prefer the ratified KHR extension and
		// enable the EXT predecessor only when KHR is unavailable.
		if (inProbe.khrCalibratedTimestamps) {
			outExtensions.pushBack(oavk::ExtKhrCalibratedTimestamps);
		} else if (inProbe.extCalibratedTimestamps) {
			outExtensions.pushBack(oavk::ExtExtCalibratedTimestamps);
		}
	}

	oa::Vector<oa::StringView> dependencies() const override {
		return {};  // Core has no dependencies
	}
};


// Factory function
oa::UniquePtr<oavk::FeatureModule> oavk::createCoreFeatures() {
	return oa::makeUnique<CoreFeatures>();
}
