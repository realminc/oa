#include "probe_shader.h"

#include <adrenotools/driver.h>
#include <adrenotools/priv.h>
#include <android/log.h>
#include <jni.h>
#include <vulkan/vulkan.h>

#include <oa/core/std.h>

#include <dlfcn.h>
#include <errno.h>
#include <sys/stat.h>

namespace {

constexpr char LogTag[] = "OA";
constexpr oa::U32 ProbeElementCount = 64;

void logInfo(const oa::String& message) {
    __android_log_print(ANDROID_LOG_INFO, LogTag, "%s", message.cStr());
}

struct ProbeError {
    oa::String message;
};

[[noreturn]] void fail(oa::String message) {
    __android_log_print(ANDROID_LOG_ERROR, LogTag, "%s", message.cStr());
    throw ProbeError{oa::move(message)};
}

void check(VkResult result, oa::StringView operation) {
    if (result != VK_SUCCESS) {
        fail(oa::format("{} failed (VkResult={})", operation, static_cast<oa::I32>(result)));
    }
}

oa::String versionString(oa::U32 version) {
    return oa::format("{}.{}.{}",
        VK_API_VERSION_MAJOR(version),
        VK_API_VERSION_MINOR(version),
        VK_API_VERSION_PATCH(version));
}

oa::String yesNo(VkBool32 value) {
    return value == VK_TRUE ? "yes" : "no";
}

oa::String bytesString(VkDeviceSize bytes) {
    static constexpr oa::Array<const char*, 4> units{"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    oa::Usize unit = 0;
    while (value >= 1024.0 && unit + 1 < units.size()) {
        value /= 1024.0;
        ++unit;
    }
    return unit == 0
        ? oa::format("{:.0f} {}", value, units[unit])
        : oa::format("{:.2f} {}", value, units[unit]);
}

class JavaString {
public:
    JavaString(JNIEnv* env, jstring value) : env_(env), value_(value) {
        chars_ = value == nullptr ? nullptr : env->GetStringUTFChars(value, nullptr);
    }

    ~JavaString() {
        if (chars_ != nullptr) {
            env_->ReleaseStringUTFChars(value_, chars_);
        }
    }

    [[nodiscard]] oa::String get() const {
        return chars_ == nullptr ? oa::String{} : oa::String(chars_);
    }

private:
    JNIEnv* env_ = nullptr;
    jstring value_ = nullptr;
    const char* chars_ = nullptr;
};

struct VulkanLibrary {
    void* handle = nullptr;
    oa::String source;

    VulkanLibrary() = default;
    VulkanLibrary(const VulkanLibrary&) = delete;
    VulkanLibrary& operator=(const VulkanLibrary&) = delete;

    VulkanLibrary(VulkanLibrary&& other) noexcept
        : handle(other.handle), source(oa::move(other.source)) {
        other.handle = nullptr;
    }

    ~VulkanLibrary() {
        if (handle != nullptr) {
            dlclose(handle);
        }
    }
};

VulkanLibrary openVulkanLibrary(
    const oa::String& source,
    oa::String driverDirectory,
    const oa::String& nativeLibraryDirectory,
    const oa::String& cacheDirectory) {
    VulkanLibrary library;
    library.source = source;

    if (source == "turnip") {
        if (!driverDirectory.empty() && driverDirectory.back() != '/') {
            driverDirectory.pushBack('/');
        }
        const oa::String temporaryDirectory = cacheDirectory + "/adrenotools";
        if (mkdir(temporaryDirectory.cStr(), 0700) != 0 and errno != EEXIST) {
            fail(oa::format(
                "Could not create temporary driver directory {} (errno={})",
                temporaryDirectory, errno));
        }
        library.handle = adrenotools_open_libvulkan(
            RTLD_NOW | RTLD_LOCAL,
            ADRENOTOOLS_DRIVER_CUSTOM,
            temporaryDirectory.cStr(),
            nativeLibraryDirectory.cStr(),
            driverDirectory.cStr(),
            "libvulkan_freedreno.so",
            nullptr,
            nullptr);
    } else {
        library.handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }

    if (library.handle == nullptr) {
        const char* error = dlerror();
        fail("Could not open " + source + " vulkan loader: "
             + (error == nullptr ? oa::String("unknown dlopen error") : oa::String(error)));
    }
    return library;
}

template <typename Function>
Function loadExport(void* library, const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    if (function == nullptr) {
        fail(oa::String("Missing vulkan export ") + name);
    }
    return function;
}

template <typename Function>
Function loadGlobal(PFN_vkGetInstanceProcAddr getInstanceProcAddr, const char* name) {
    auto function = reinterpret_cast<Function>(getInstanceProcAddr(VK_NULL_HANDLE, name));
    if (function == nullptr) {
        fail(oa::String("Missing vulkan global function ") + name);
    }
    return function;
}

template <typename Function>
Function loadInstance(
    PFN_vkGetInstanceProcAddr getInstanceProcAddr,
    VkInstance instance,
    const char* name) {
    auto function = reinterpret_cast<Function>(getInstanceProcAddr(instance, name));
    if (function == nullptr) {
        fail(oa::String("Missing vulkan instance function ") + name);
    }
    return function;
}

template <typename Function>
Function loadDevice(PFN_vkGetDeviceProcAddr getDeviceProcAddr, VkDevice device, const char* name) {
    auto function = reinterpret_cast<Function>(getDeviceProcAddr(device, name));
    if (function == nullptr) {
        fail(oa::String("Missing vulkan device function ") + name);
    }
    return function;
}

struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;
    PFN_vkGetDeviceProcAddr getDeviceProcAddr = nullptr;
    PFN_vkEnumerateInstanceVersion enumerateInstanceVersion = nullptr;
    PFN_vkCreateInstance createInstance = nullptr;
    PFN_vkDestroyInstance destroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices enumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties getPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 getPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 getPhysicalDeviceFeatures2 = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties enumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties getPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties getPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice createDevice = nullptr;
};

struct DeviceDispatch {
    PFN_vkDestroyDevice destroyDevice = nullptr;
    PFN_vkGetDeviceQueue getDeviceQueue = nullptr;
    PFN_vkCreateBuffer createBuffer = nullptr;
    PFN_vkDestroyBuffer destroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory allocateMemory = nullptr;
    PFN_vkFreeMemory freeMemory = nullptr;
    PFN_vkBindBufferMemory bindBufferMemory = nullptr;
    PFN_vkMapMemory mapMemory = nullptr;
    PFN_vkUnmapMemory unmapMemory = nullptr;
    PFN_vkFlushMappedMemoryRanges flushMappedMemoryRanges = nullptr;
    PFN_vkInvalidateMappedMemoryRanges invalidateMappedMemoryRanges = nullptr;
    PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool createDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool destroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
    PFN_vkCreateShaderModule createShaderModule = nullptr;
    PFN_vkDestroyShaderModule destroyShaderModule = nullptr;
    PFN_vkCreatePipelineLayout createPipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout destroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines createComputePipelines = nullptr;
    PFN_vkDestroyPipeline destroyPipeline = nullptr;
    PFN_vkCreateCommandPool createCommandPool = nullptr;
    PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCreateFence createFence = nullptr;
    PFN_vkDestroyFence destroyFence = nullptr;
    PFN_vkQueueSubmit queueSubmit = nullptr;
    PFN_vkWaitForFences waitForFences = nullptr;
};

DeviceDispatch loadDeviceDispatch(
    PFN_vkGetDeviceProcAddr getDeviceProcAddr,
    VkDevice device) {
    DeviceDispatch functions;
#define OA_LOAD_DEVICE(member, name) \
    functions.member = loadDevice<PFN_##name>(getDeviceProcAddr, device, #name)
    OA_LOAD_DEVICE(destroyDevice, vkDestroyDevice);
    OA_LOAD_DEVICE(getDeviceQueue, vkGetDeviceQueue);
    OA_LOAD_DEVICE(createBuffer, vkCreateBuffer);
    OA_LOAD_DEVICE(destroyBuffer, vkDestroyBuffer);
    OA_LOAD_DEVICE(getBufferMemoryRequirements, vkGetBufferMemoryRequirements);
    OA_LOAD_DEVICE(allocateMemory, vkAllocateMemory);
    OA_LOAD_DEVICE(freeMemory, vkFreeMemory);
    OA_LOAD_DEVICE(bindBufferMemory, vkBindBufferMemory);
    OA_LOAD_DEVICE(mapMemory, vkMapMemory);
    OA_LOAD_DEVICE(unmapMemory, vkUnmapMemory);
    OA_LOAD_DEVICE(flushMappedMemoryRanges, vkFlushMappedMemoryRanges);
    OA_LOAD_DEVICE(invalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges);
    OA_LOAD_DEVICE(createDescriptorSetLayout, vkCreateDescriptorSetLayout);
    OA_LOAD_DEVICE(destroyDescriptorSetLayout, vkDestroyDescriptorSetLayout);
    OA_LOAD_DEVICE(createDescriptorPool, vkCreateDescriptorPool);
    OA_LOAD_DEVICE(destroyDescriptorPool, vkDestroyDescriptorPool);
    OA_LOAD_DEVICE(allocateDescriptorSets, vkAllocateDescriptorSets);
    OA_LOAD_DEVICE(updateDescriptorSets, vkUpdateDescriptorSets);
    OA_LOAD_DEVICE(createShaderModule, vkCreateShaderModule);
    OA_LOAD_DEVICE(destroyShaderModule, vkDestroyShaderModule);
    OA_LOAD_DEVICE(createPipelineLayout, vkCreatePipelineLayout);
    OA_LOAD_DEVICE(destroyPipelineLayout, vkDestroyPipelineLayout);
    OA_LOAD_DEVICE(createComputePipelines, vkCreateComputePipelines);
    OA_LOAD_DEVICE(destroyPipeline, vkDestroyPipeline);
    OA_LOAD_DEVICE(createCommandPool, vkCreateCommandPool);
    OA_LOAD_DEVICE(destroyCommandPool, vkDestroyCommandPool);
    OA_LOAD_DEVICE(allocateCommandBuffers, vkAllocateCommandBuffers);
    OA_LOAD_DEVICE(beginCommandBuffer, vkBeginCommandBuffer);
    OA_LOAD_DEVICE(endCommandBuffer, vkEndCommandBuffer);
    OA_LOAD_DEVICE(cmdPipelineBarrier, vkCmdPipelineBarrier);
    OA_LOAD_DEVICE(cmdBindPipeline, vkCmdBindPipeline);
    OA_LOAD_DEVICE(cmdBindDescriptorSets, vkCmdBindDescriptorSets);
    OA_LOAD_DEVICE(cmdDispatch, vkCmdDispatch);
    OA_LOAD_DEVICE(createFence, vkCreateFence);
    OA_LOAD_DEVICE(destroyFence, vkDestroyFence);
    OA_LOAD_DEVICE(queueSubmit, vkQueueSubmit);
    OA_LOAD_DEVICE(waitForFences, vkWaitForFences);
#undef OA_LOAD_DEVICE
    return functions;
}

bool hasExtension(const oa::Vector<VkExtensionProperties>& extensions, const char* name) {
    for (const VkExtensionProperties& extension : extensions) {
        if (oa::strcmp(extension.extensionName, name) == 0) return true;
    }
    return false;
}

struct FeatureCaps {
    VkPhysicalDeviceVulkan11Features core11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features core12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features core13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures extBda{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures extDescriptor{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceTimelineSemaphoreFeatures extTimeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceSynchronization2Features extSync2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceDynamicRenderingFeatures extDynamicRendering{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
};

FeatureCaps queryFeatureCaps(
    const InstanceDispatch& vk,
    VkPhysicalDevice physicalDevice,
    oa::U32 deviceApi,
    const oa::Vector<VkExtensionProperties>& extensions) {
    FeatureCaps caps;

    VkPhysicalDeviceFeatures2 core{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    core.pNext = &caps.core11;
    caps.core11.pNext = &caps.core12;
    caps.core12.pNext = &caps.core13;
    vk.getPhysicalDeviceFeatures2(physicalDevice, &core);

    VkPhysicalDeviceFeatures2 extensionFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void** tail = &extensionFeatures.pNext;
    const auto append = [&tail](auto& feature) {
        *tail = &feature;
        tail = &feature.pNext;
    };

    if (deviceApi < VK_API_VERSION_1_2) {
        if (hasExtension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
            append(caps.extBda);
        }
        if (hasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            append(caps.extDescriptor);
        }
        if (hasExtension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
            append(caps.extTimeline);
        }
    }
    if (deviceApi < VK_API_VERSION_1_3) {
        if (hasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
            append(caps.extSync2);
        }
        if (hasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
            append(caps.extDynamicRendering);
        }
    }
    if (extensionFeatures.pNext != nullptr) {
        vk.getPhysicalDeviceFeatures2(physicalDevice, &extensionFeatures);
    }
    return caps;
}

oa::U32 findHostMemoryType(
    const VkPhysicalDeviceMemoryProperties& memory,
    oa::U32 allowedTypes,
    bool& coherent) {
    oa::U32 fallback = UINT32_MAX;
    for (oa::U32 index = 0; index < memory.memoryTypeCount; ++index) {
        if ((allowedTypes & (1u << index)) == 0) {
            continue;
        }
        const VkMemoryPropertyFlags flags = memory.memoryTypes[index].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
            continue;
        }
        if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
            coherent = true;
            return index;
        }
        fallback = index;
    }
    coherent = false;
    if (fallback == UINT32_MAX) {
        fail("No host-visible vulkan memory type for probe buffer");
    }
    return fallback;
}

oa::String runComputeDispatch(
    const InstanceDispatch& instanceFunctions,
    VkPhysicalDevice physicalDevice,
    oa::U32 computeQueueFamily) {
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = computeQueueFamily,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
    };

    VkDevice device = VK_NULL_HANDLE;
    check(instanceFunctions.createDevice(physicalDevice, &deviceInfo, nullptr, &device),
          "vkCreateDevice(minimal probe)");
    DeviceDispatch vk = loadDeviceDispatch(instanceFunctions.getDeviceProcAddr, device);

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    void* mapped = nullptr;

    try {
        constexpr VkDeviceSize bufferSize = ProbeElementCount * sizeof(oa::U32);
        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        check(vk.createBuffer(device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vk.getBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        instanceFunctions.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        bool coherent = false;
        const oa::U32 memoryType = findHostMemoryType(
            memoryProperties, requirements.memoryTypeBits, coherent);

        VkMemoryAllocateInfo allocationInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = memoryType,
        };
        check(vk.allocateMemory(device, &allocationInfo, nullptr, &memory), "vkAllocateMemory");
        check(vk.bindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        check(vk.mapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");

        auto* values = static_cast<oa::U32*>(mapped);
        for (oa::U32 index = 0; index < ProbeElementCount; ++index) {
            values[index] = index;
        }
        if (!coherent) {
            VkMappedMemoryRange range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = memory,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            check(vk.flushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
        }

        VkDescriptorSetLayoutBinding binding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
        VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        check(vk.createDescriptorSetLayout(
                  device, &descriptorLayoutInfo, nullptr, &descriptorLayout),
              "vkCreateDescriptorSetLayout");

        VkDescriptorPoolSize poolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
        };
        VkDescriptorPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &poolSize,
        };
        check(vk.createDescriptorPool(device, &poolInfo, nullptr, &descriptorPool),
              "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo setInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptorLayout,
        };
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        check(vk.allocateDescriptorSets(device, &setInfo, &descriptorSet),
              "vkAllocateDescriptorSets");
        VkDescriptorBufferInfo descriptorBuffer{
            .buffer = buffer,
            .offset = 0,
            .range = bufferSize,
        };
        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &descriptorBuffer,
        };
        vk.updateDescriptorSets(device, 1, &write, 0, nullptr);

        VkShaderModuleCreateInfo shaderInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = OaProbeShaderSpirvSize,
            .pCode = OaProbeShaderSpirv,
        };
        check(vk.createShaderModule(device, &shaderInfo, nullptr, &shader),
              "vkCreateShaderModule");
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorLayout,
        };
        check(vk.createPipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
              "vkCreatePipelineLayout");
        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        };
        VkComputePipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = stage,
            .layout = pipelineLayout,
        };
        check(vk.createComputePipelines(
                  device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
              "vkCreateComputePipelines");

        VkCommandPoolCreateInfo commandPoolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = computeQueueFamily,
        };
        check(vk.createCommandPool(device, &commandPoolInfo, nullptr, &commandPool),
              "vkCreateCommandPool");
        VkCommandBufferAllocateInfo commandBufferInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        check(vk.allocateCommandBuffers(device, &commandBufferInfo, &commandBuffer),
              "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        check(vk.beginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

        VkBufferMemoryBarrier hostToCompute{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vk.cmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            nullptr,
            1,
            &hostToCompute,
            0,
            nullptr);
        vk.cmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vk.cmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);
        vk.cmdDispatch(commandBuffer, 1, 1, 1);

        VkBufferMemoryBarrier computeToHost{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vk.cmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            0,
            nullptr,
            1,
            &computeToHost,
            0,
            nullptr);
        check(vk.endCommandBuffer(commandBuffer), "vkEndCommandBuffer");

        VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        check(vk.createFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");
        VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };
        VkQueue queue = VK_NULL_HANDLE;
        vk.getDeviceQueue(device, computeQueueFamily, 0, &queue);
        check(vk.queueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
        check(vk.waitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ULL),
              "vkWaitForFences");

        if (!coherent) {
            VkMappedMemoryRange range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = memory,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            check(vk.invalidateMappedMemoryRanges(device, 1, &range),
                  "vkInvalidateMappedMemoryRanges");
        }

        for (oa::U32 index = 0; index < ProbeElementCount; ++index) {
            const oa::U32 expected = index * 3u + 7u;
            if (values[index] != expected) {
                fail(oa::format(
                    "Compute result mismatch at index {}: expected {}, got {}",
                    index, expected, values[index]));
            }
        }

        vk.destroyFence(device, fence, nullptr);
        vk.destroyCommandPool(device, commandPool, nullptr);
        vk.destroyPipeline(device, pipeline, nullptr);
        vk.destroyPipelineLayout(device, pipelineLayout, nullptr);
        vk.destroyShaderModule(device, shader, nullptr);
        vk.destroyDescriptorPool(device, descriptorPool, nullptr);
        vk.destroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        vk.unmapMemory(device, memory);
        vk.freeMemory(device, memory, nullptr);
        vk.destroyBuffer(device, buffer, nullptr);
        vk.destroyDevice(device, nullptr);
        return coherent ? "PASS (64 values, coherent host memory)"
                        : "PASS (64 values, explicit cache management)";
    } catch (...) {
        if (fence != VK_NULL_HANDLE) vk.destroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vk.destroyCommandPool(device, commandPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vk.destroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vk.destroyPipelineLayout(device, pipelineLayout, nullptr);
        if (shader != VK_NULL_HANDLE) vk.destroyShaderModule(device, shader, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vk.destroyDescriptorPool(device, descriptorPool, nullptr);
        if (descriptorLayout != VK_NULL_HANDLE) {
            vk.destroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        }
        if (mapped != nullptr) vk.unmapMemory(device, memory);
        if (memory != VK_NULL_HANDLE) vk.freeMemory(device, memory, nullptr);
        if (buffer != VK_NULL_HANDLE) vk.destroyBuffer(device, buffer, nullptr);
        vk.destroyDevice(device, nullptr);
        throw;
    }
}

oa::String buildReport(
    const oa::String& source,
    const oa::String& driverDirectory,
    const oa::String& nativeLibraryDirectory,
    const oa::String& cacheDirectory) {
    VulkanLibrary library = openVulkanLibrary(
        source, driverDirectory, nativeLibraryDirectory, cacheDirectory);

    InstanceDispatch vk;
    vk.getInstanceProcAddr = loadExport<PFN_vkGetInstanceProcAddr>(
        library.handle, "vkGetInstanceProcAddr");
    vk.enumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vk.getInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    vk.createInstance = loadGlobal<PFN_vkCreateInstance>(
        vk.getInstanceProcAddr, "vkCreateInstance");

    oa::U32 loaderApi = VK_API_VERSION_1_0;
    if (vk.enumerateInstanceVersion != nullptr) {
        check(vk.enumerateInstanceVersion(&loaderApi), "vkEnumerateInstanceVersion");
    }
    const oa::U32 requestedApi = oa::min(loaderApi, VK_API_VERSION_1_3);
    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "OA Mobile vulkan probe",
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "OA",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = requestedApi,
    };
    VkInstanceCreateInfo instanceInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &applicationInfo,
    };
    VkInstance instance = VK_NULL_HANDLE;
    check(vk.createInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");

    try {
#define OA_LOAD_INSTANCE(member, name) \
        vk.member = loadInstance<PFN_##name>(vk.getInstanceProcAddr, instance, #name)
        OA_LOAD_INSTANCE(destroyInstance, vkDestroyInstance);
        OA_LOAD_INSTANCE(enumeratePhysicalDevices, vkEnumeratePhysicalDevices);
        OA_LOAD_INSTANCE(getPhysicalDeviceProperties, vkGetPhysicalDeviceProperties);
        OA_LOAD_INSTANCE(getPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties2);
        OA_LOAD_INSTANCE(getPhysicalDeviceFeatures2, vkGetPhysicalDeviceFeatures2);
        OA_LOAD_INSTANCE(enumerateDeviceExtensionProperties, vkEnumerateDeviceExtensionProperties);
        OA_LOAD_INSTANCE(getPhysicalDeviceQueueFamilyProperties, vkGetPhysicalDeviceQueueFamilyProperties);
        OA_LOAD_INSTANCE(getPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties);
        OA_LOAD_INSTANCE(createDevice, vkCreateDevice);
        OA_LOAD_INSTANCE(getDeviceProcAddr, vkGetDeviceProcAddr);
#undef OA_LOAD_INSTANCE

        oa::U32 physicalDeviceCount = 0;
        check(vk.enumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr),
              "vkEnumeratePhysicalDevices(count)");
        if (physicalDeviceCount == 0) {
            fail("vulkan loader returned zero physical devices");
        }
        oa::Vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        check(vk.enumeratePhysicalDevices(
                  instance, &physicalDeviceCount, physicalDevices.data()),
              "vkEnumeratePhysicalDevices(data)");
        VkPhysicalDevice physicalDevice = physicalDevices.front();

        VkPhysicalDeviceProperties properties{};
        vk.getPhysicalDeviceProperties(physicalDevice, &properties);

        oa::U32 extensionCount = 0;
        check(vk.enumerateDeviceExtensionProperties(
                  physicalDevice, nullptr, &extensionCount, nullptr),
              "vkEnumerateDeviceExtensionProperties(count)");
        oa::Vector<VkExtensionProperties> extensions(extensionCount);
        check(vk.enumerateDeviceExtensionProperties(
                  physicalDevice, nullptr, &extensionCount, extensions.data()),
              "vkEnumerateDeviceExtensionProperties(data)");

        VkPhysicalDeviceSubgroupProperties subgroup{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceDriverProperties driver{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceDescriptorIndexingProperties descriptorLimits{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
        VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudget{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 properties2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties2.pNext = &subgroup;
        void** propertyTail = &subgroup.pNext;
        if (properties.apiVersion >= VK_API_VERSION_1_2
            || hasExtension(extensions, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
            *propertyTail = &driver;
            propertyTail = &driver.pNext;
        }
        if (properties.apiVersion >= VK_API_VERSION_1_2
            || hasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            *propertyTail = &descriptorLimits;
            propertyTail = &descriptorLimits.pNext;
        }
        if (hasExtension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            *propertyTail = &memoryBudget;
        }
        vk.getPhysicalDeviceProperties2(physicalDevice, &properties2);

        FeatureCaps features = queryFeatureCaps(
            vk, physicalDevice, properties.apiVersion, extensions);
        const bool core12 = properties.apiVersion >= VK_API_VERSION_1_2;
        const bool core13 = properties.apiVersion >= VK_API_VERSION_1_3;
        const bool bufferDeviceAddress = core12
            ? features.core12.bufferDeviceAddress == VK_TRUE
            : features.extBda.bufferDeviceAddress == VK_TRUE;
        const bool descriptorIndexing = core12
            ? features.core12.descriptorIndexing == VK_TRUE
            : hasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        const auto descriptorValue = [core12](VkBool32 core, VkBool32 extension) {
            return core12 ? core : extension;
        };
        const bool timelineSemaphore = core12
            ? features.core12.timelineSemaphore == VK_TRUE
            : features.extTimeline.timelineSemaphore == VK_TRUE;
        const bool synchronization2 = core13
            ? features.core13.synchronization2 == VK_TRUE
            : features.extSync2.synchronization2 == VK_TRUE;
        const bool dynamicRendering = core13
            ? features.core13.dynamicRendering == VK_TRUE
            : features.extDynamicRendering.dynamicRendering == VK_TRUE;

        struct Requirement {
            const char* name;
            bool supported;
        };
        const oa::Array<Requirement, 14> requirements{
            Requirement{"bufferDeviceAddress", bufferDeviceAddress},
            Requirement{"descriptorIndexing", descriptorIndexing},
            Requirement{"runtimeDescriptorArray", descriptorValue(
                features.core12.runtimeDescriptorArray,
                features.extDescriptor.runtimeDescriptorArray) == VK_TRUE},
            Requirement{"descriptorBindingPartiallyBound", descriptorValue(
                features.core12.descriptorBindingPartiallyBound,
                features.extDescriptor.descriptorBindingPartiallyBound) == VK_TRUE},
            Requirement{"descriptorBindingVariableDescriptorCount", descriptorValue(
                features.core12.descriptorBindingVariableDescriptorCount,
                features.extDescriptor.descriptorBindingVariableDescriptorCount) == VK_TRUE},
            Requirement{"shaderSampledImageArrayNonUniformIndexing", descriptorValue(
                features.core12.shaderSampledImageArrayNonUniformIndexing,
                features.extDescriptor.shaderSampledImageArrayNonUniformIndexing) == VK_TRUE},
            Requirement{"shaderStorageBufferArrayNonUniformIndexing", descriptorValue(
                features.core12.shaderStorageBufferArrayNonUniformIndexing,
                features.extDescriptor.shaderStorageBufferArrayNonUniformIndexing) == VK_TRUE},
            Requirement{"descriptorBindingStorageBufferUpdateAfterBind", descriptorValue(
                features.core12.descriptorBindingStorageBufferUpdateAfterBind,
                features.extDescriptor.descriptorBindingStorageBufferUpdateAfterBind) == VK_TRUE},
            Requirement{"descriptorBindingStorageImageUpdateAfterBind", descriptorValue(
                features.core12.descriptorBindingStorageImageUpdateAfterBind,
                features.extDescriptor.descriptorBindingStorageImageUpdateAfterBind) == VK_TRUE},
            Requirement{"descriptorBindingSampledImageUpdateAfterBind", descriptorValue(
                features.core12.descriptorBindingSampledImageUpdateAfterBind,
                features.extDescriptor.descriptorBindingSampledImageUpdateAfterBind) == VK_TRUE},
            Requirement{"descriptorBindingUpdateUnusedWhilePending", descriptorValue(
                features.core12.descriptorBindingUpdateUnusedWhilePending,
                features.extDescriptor.descriptorBindingUpdateUnusedWhilePending) == VK_TRUE},
            Requirement{"timelineSemaphore", timelineSemaphore},
            Requirement{"synchronization2", synchronization2},
            Requirement{"dynamicRendering (current OA Core)", dynamicRendering},
        };
        bool profilePass = true;
        for (const Requirement& requirement : requirements) {
            profilePass = profilePass and requirement.supported;
        }

        oa::U32 queueFamilyCount = 0;
        vk.getPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueFamilyCount, nullptr);
        oa::Vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vk.getPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueFamilyCount, queueFamilies.data());
        oa::U32 computeQueueFamily = UINT32_MAX;
        for (oa::U32 index = 0; index < queueFamilyCount; ++index) {
            if ((queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                computeQueueFamily = index;
                break;
            }
        }
        if (computeQueueFamily == UINT32_MAX) {
            fail("No vulkan compute queue family");
        }

        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vk.getPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        oa::String dispatchResult;
        try {
            dispatchResult = runComputeDispatch(vk, physicalDevice, computeQueueFamily);
        } catch (const ProbeError& error) {
            dispatchResult = oa::String("FAIL (") + error.message + ")";
        }

        oa::String report = oa::format(
            "OA MOBILE LAB / VULKAN PROBE\n"
            "================================\n"
            "Driver source: {}\n"
            "Loader API:    {}\n"
            "instance API:  {}\n"
            "Device:        {}\n"
            "Device API:    {}\n"
            "Vendor/Device: 0x{:x} / 0x{:x}\n"
            "Driver ID:     {}\n"
            "Driver name:   {}\n"
            "Driver info:   {}\n"
            "Conformance:   {}.{}.{}.{}\n"
            "Extensions:    {}\n"
            "queue family:  {} (count={}, timestampBits={})\n"
            "Subgroup:      size={}, ops=0x{:x}\n\n"
            "OA ModernCompute contract\n"
            "-------------------------\n"
            "ModernCompute: {}\n"
            "Core 1.3 ABI:   {}\n",
            source == "turnip" ? "Bundled Turnip" : "System Adreno",
            versionString(loaderApi), versionString(requestedApi), properties.deviceName,
            versionString(properties.apiVersion), properties.vendorID, properties.deviceID,
            static_cast<oa::U32>(driver.driverID),
            driver.driverName[0] == '\0' ? "(not reported)" : driver.driverName,
            driver.driverInfo[0] == '\0' ? "(not reported)" : driver.driverInfo,
            static_cast<oa::U32>(driver.conformanceVersion.major),
            static_cast<oa::U32>(driver.conformanceVersion.minor),
            static_cast<oa::U32>(driver.conformanceVersion.subminor),
            static_cast<oa::U32>(driver.conformanceVersion.patch),
            extensions.size(), computeQueueFamily,
            queueFamilies[computeQueueFamily].queueCount,
            queueFamilies[computeQueueFamily].timestampValidBits,
            subgroup.subgroupSize, subgroup.supportedOperations,
            profilePass ? "PASS" : "FAIL",
            core13 ? "yes" : "no — extension adapters required");
        for (const Requirement& requirement : requirements) {
            report += oa::format("{}{}\n",
                requirement.supported ? "[+] " : "[-] ", requirement.name);
        }

        const char* bdaProvider = core12 ? "core 1.2" :
            hasExtension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
                ? "VK_KHR_buffer_device_address" : "missing";
        const char* descriptorProvider = core12 ? "core 1.2" :
            hasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
                ? "VK_EXT_descriptor_indexing" : "missing";
        const char* timelineProvider = core12 ? "core 1.2" :
            hasExtension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)
                ? "VK_KHR_timeline_semaphore" : "missing";
        const char* sync2Provider = core13 ? "core 1.3" :
            hasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
                ? "VK_KHR_synchronization2" : "missing";
        const char* dynamicProvider = core13 ? "core 1.3" :
            hasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
                ? "VK_KHR_dynamic_rendering" : "missing";
        report += oa::format(
            "\nFeature detail\n"
            "--------------\n"
            "16-bit storage:       {}\n"
            "shaderFloat16:        {}\n"
            "shaderInt8:           {}\n"
            "8-bit storage:        {}\n"
            "BDA provider:         {}\n"
            "Descriptor provider:  {}\n"
            "Timeline provider:    {}\n"
            "Sync2 provider:       {}\n"
            "dynamic provider:     {}\n"
            "\nDescriptor limits\n"
            "-----------------\n"
            "UAB storage buffers/stage: {}\n"
            "UAB sampled images/stage:  {}\n"
            "max storage buffer range:  {}\n"
            "\nMemory heaps\n"
            "------------\n",
            yesNo(features.core11.storageBuffer16BitAccess),
            yesNo(features.core12.shaderFloat16), yesNo(features.core12.shaderInt8),
            yesNo(features.core12.storageBuffer8BitAccess), bdaProvider,
            descriptorProvider, timelineProvider, sync2Provider, dynamicProvider,
            descriptorLimits.maxPerStageDescriptorUpdateAfterBindStorageBuffers,
            descriptorLimits.maxPerStageDescriptorUpdateAfterBindSampledImages,
            bytesString(properties.limits.maxStorageBufferRange));
        for (oa::U32 heap = 0; heap < memoryProperties.memoryHeapCount; ++heap) {
            report += oa::format("heap {}: {}", heap,
                bytesString(memoryProperties.memoryHeaps[heap].size));
            if ((memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                report += " device-local";
            }
            if (hasExtension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
                report += oa::format(", budget={}, usage={}",
                    bytesString(memoryBudget.heapBudget[heap]),
                    bytesString(memoryBudget.heapUsage[heap]));
            }
            report += '\n';
        }

        report += oa::format(
            "\nExecution proof\n"
            "---------------\n"
            "shader:   values[i] = values[i] * 3 + 7\n"
            "Dispatch: {}\n", dispatchResult);

        logInfo("vulkan probe complete: source=" + source
                + ", device=" + oa::String(properties.deviceName)
                + ", API=" + versionString(properties.apiVersion)
                + ", ModernCompute=" + (profilePass ? "PASS" : "FAIL")
                + ", Dispatch=" + dispatchResult);

        vk.destroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
        return report;
    } catch (...) {
        if (instance != VK_NULL_HANDLE && vk.destroyInstance != nullptr) {
            vk.destroyInstance(instance, nullptr);
        }
        throw;
    }
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_oa_mobilelab_ProbeService_nativeRunProbe(
    JNIEnv* env,
    jclass,
    jstring driverSource,
    jstring driverDirectory,
    jstring nativeLibraryDirectory,
    jstring cacheDirectory) {
    const oa::String source = JavaString(env, driverSource).get();
    const oa::String drivers = JavaString(env, driverDirectory).get();
    const oa::String nativeLibraries = JavaString(env, nativeLibraryDirectory).get();
    const oa::String cache = JavaString(env, cacheDirectory).get();

    try {
        const oa::String report = buildReport(source, drivers, nativeLibraries, cache);
        return env->NewStringUTF(report.cStr());
    } catch (const ProbeError& error) {
        const oa::String report =
            "OA MOBILE LAB / VULKAN PROBE\n"
            "================================\n"
            "Driver source: " + source + "\n"
            "Fatal: " + error.message + "\n";
        return env->NewStringUTF(report.cStr());
    }
}
