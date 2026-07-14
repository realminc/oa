#include "probe_shader.h"

#include <adrenotools/driver.h>
#include <adrenotools/priv.h>
#include <android/log.h>
#include <jni.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr char LogTag[] = "OA";
constexpr std::uint32_t ProbeElementCount = 64;

void LogInfo(const std::string& message) {
    __android_log_print(ANDROID_LOG_INFO, LogTag, "%s", message.c_str());
}

[[noreturn]] void Fail(const std::string& message) {
    __android_log_print(ANDROID_LOG_ERROR, LogTag, "%s", message.c_str());
    throw std::runtime_error(message);
}

void Check(VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        Fail(std::string(operation) + " failed (VkResult="
             + std::to_string(static_cast<int>(result)) + ")");
    }
}

std::string VersionString(std::uint32_t version) {
    std::ostringstream out;
    out << VK_API_VERSION_MAJOR(version) << '.'
        << VK_API_VERSION_MINOR(version) << '.'
        << VK_API_VERSION_PATCH(version);
    return out.str();
}

std::string YesNo(VkBool32 value) {
    return value == VK_TRUE ? "yes" : "no";
}

std::string BytesString(VkDeviceSize bytes) {
    static constexpr std::array<const char*, 4> units{"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < units.size()) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << ' ' << units[unit];
    return out.str();
}

class JavaString {
public:
    JavaString(JNIEnv* env, jstring value) : Env_(env), Value_(value) {
        Chars_ = value == nullptr ? nullptr : env->GetStringUTFChars(value, nullptr);
    }

    ~JavaString() {
        if (Chars_ != nullptr) {
            Env_->ReleaseStringUTFChars(Value_, Chars_);
        }
    }

    [[nodiscard]] std::string Get() const {
        return Chars_ == nullptr ? std::string{} : std::string(Chars_);
    }

private:
    JNIEnv* Env_ = nullptr;
    jstring Value_ = nullptr;
    const char* Chars_ = nullptr;
};

struct VulkanLibrary {
    void* Handle = nullptr;
    std::string Source;

    VulkanLibrary() = default;
    VulkanLibrary(const VulkanLibrary&) = delete;
    VulkanLibrary& operator=(const VulkanLibrary&) = delete;

    VulkanLibrary(VulkanLibrary&& other) noexcept
        : Handle(other.Handle), Source(std::move(other.Source)) {
        other.Handle = nullptr;
    }

    ~VulkanLibrary() {
        if (Handle != nullptr) {
            dlclose(Handle);
        }
    }
};

VulkanLibrary OpenVulkanLibrary(
    const std::string& source,
    std::string driverDirectory,
    const std::string& nativeLibraryDirectory,
    const std::string& cacheDirectory) {
    VulkanLibrary library;
    library.Source = source;

    if (source == "turnip") {
        if (!driverDirectory.empty() && driverDirectory.back() != '/') {
            driverDirectory.push_back('/');
        }
        const std::string temporaryDirectory = cacheDirectory + "/adrenotools";
        std::filesystem::create_directories(temporaryDirectory);
        library.Handle = adrenotools_open_libvulkan(
            RTLD_NOW | RTLD_LOCAL,
            ADRENOTOOLS_DRIVER_CUSTOM,
            temporaryDirectory.c_str(),
            nativeLibraryDirectory.c_str(),
            driverDirectory.c_str(),
            "libvulkan_freedreno.so",
            nullptr,
            nullptr);
    } else {
        library.Handle = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }

    if (library.Handle == nullptr) {
        const char* error = dlerror();
        Fail("Could not open " + source + " Vulkan loader: "
             + (error == nullptr ? std::string("unknown dlopen error") : std::string(error)));
    }
    return library;
}

template <typename Function>
Function LoadExport(void* library, const char* name) {
    auto function = reinterpret_cast<Function>(dlsym(library, name));
    if (function == nullptr) {
        Fail(std::string("Missing Vulkan export ") + name);
    }
    return function;
}

template <typename Function>
Function LoadGlobal(PFN_vkGetInstanceProcAddr getInstanceProcAddr, const char* name) {
    auto function = reinterpret_cast<Function>(getInstanceProcAddr(VK_NULL_HANDLE, name));
    if (function == nullptr) {
        Fail(std::string("Missing Vulkan global function ") + name);
    }
    return function;
}

template <typename Function>
Function LoadInstance(
    PFN_vkGetInstanceProcAddr getInstanceProcAddr,
    VkInstance instance,
    const char* name) {
    auto function = reinterpret_cast<Function>(getInstanceProcAddr(instance, name));
    if (function == nullptr) {
        Fail(std::string("Missing Vulkan instance function ") + name);
    }
    return function;
}

template <typename Function>
Function LoadDevice(PFN_vkGetDeviceProcAddr getDeviceProcAddr, VkDevice device, const char* name) {
    auto function = reinterpret_cast<Function>(getDeviceProcAddr(device, name));
    if (function == nullptr) {
        Fail(std::string("Missing Vulkan device function ") + name);
    }
    return function;
}

struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = nullptr;
    PFN_vkGetDeviceProcAddr GetDeviceProcAddr = nullptr;
    PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion = nullptr;
    PFN_vkCreateInstance CreateInstance = nullptr;
    PFN_vkDestroyInstance DestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2 = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice CreateDevice = nullptr;
};

struct DeviceDispatch {
    PFN_vkDestroyDevice DestroyDevice = nullptr;
    PFN_vkGetDeviceQueue GetDeviceQueue = nullptr;
    PFN_vkCreateBuffer CreateBuffer = nullptr;
    PFN_vkDestroyBuffer DestroyBuffer = nullptr;
    PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements = nullptr;
    PFN_vkAllocateMemory AllocateMemory = nullptr;
    PFN_vkFreeMemory FreeMemory = nullptr;
    PFN_vkBindBufferMemory BindBufferMemory = nullptr;
    PFN_vkMapMemory MapMemory = nullptr;
    PFN_vkUnmapMemory UnmapMemory = nullptr;
    PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges = nullptr;
    PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges = nullptr;
    PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout = nullptr;
    PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout = nullptr;
    PFN_vkCreateDescriptorPool CreateDescriptorPool = nullptr;
    PFN_vkDestroyDescriptorPool DestroyDescriptorPool = nullptr;
    PFN_vkAllocateDescriptorSets AllocateDescriptorSets = nullptr;
    PFN_vkUpdateDescriptorSets UpdateDescriptorSets = nullptr;
    PFN_vkCreateShaderModule CreateShaderModule = nullptr;
    PFN_vkDestroyShaderModule DestroyShaderModule = nullptr;
    PFN_vkCreatePipelineLayout CreatePipelineLayout = nullptr;
    PFN_vkDestroyPipelineLayout DestroyPipelineLayout = nullptr;
    PFN_vkCreateComputePipelines CreateComputePipelines = nullptr;
    PFN_vkDestroyPipeline DestroyPipeline = nullptr;
    PFN_vkCreateCommandPool CreateCommandPool = nullptr;
    PFN_vkDestroyCommandPool DestroyCommandPool = nullptr;
    PFN_vkAllocateCommandBuffers AllocateCommandBuffers = nullptr;
    PFN_vkBeginCommandBuffer BeginCommandBuffer = nullptr;
    PFN_vkEndCommandBuffer EndCommandBuffer = nullptr;
    PFN_vkCmdPipelineBarrier CmdPipelineBarrier = nullptr;
    PFN_vkCmdBindPipeline CmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets = nullptr;
    PFN_vkCmdDispatch CmdDispatch = nullptr;
    PFN_vkCreateFence CreateFence = nullptr;
    PFN_vkDestroyFence DestroyFence = nullptr;
    PFN_vkQueueSubmit QueueSubmit = nullptr;
    PFN_vkWaitForFences WaitForFences = nullptr;
};

DeviceDispatch LoadDeviceDispatch(
    PFN_vkGetDeviceProcAddr getDeviceProcAddr,
    VkDevice device) {
    DeviceDispatch functions;
#define OA_LOAD_DEVICE(member, name) \
    functions.member = LoadDevice<PFN_##name>(getDeviceProcAddr, device, #name)
    OA_LOAD_DEVICE(DestroyDevice, vkDestroyDevice);
    OA_LOAD_DEVICE(GetDeviceQueue, vkGetDeviceQueue);
    OA_LOAD_DEVICE(CreateBuffer, vkCreateBuffer);
    OA_LOAD_DEVICE(DestroyBuffer, vkDestroyBuffer);
    OA_LOAD_DEVICE(GetBufferMemoryRequirements, vkGetBufferMemoryRequirements);
    OA_LOAD_DEVICE(AllocateMemory, vkAllocateMemory);
    OA_LOAD_DEVICE(FreeMemory, vkFreeMemory);
    OA_LOAD_DEVICE(BindBufferMemory, vkBindBufferMemory);
    OA_LOAD_DEVICE(MapMemory, vkMapMemory);
    OA_LOAD_DEVICE(UnmapMemory, vkUnmapMemory);
    OA_LOAD_DEVICE(FlushMappedMemoryRanges, vkFlushMappedMemoryRanges);
    OA_LOAD_DEVICE(InvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges);
    OA_LOAD_DEVICE(CreateDescriptorSetLayout, vkCreateDescriptorSetLayout);
    OA_LOAD_DEVICE(DestroyDescriptorSetLayout, vkDestroyDescriptorSetLayout);
    OA_LOAD_DEVICE(CreateDescriptorPool, vkCreateDescriptorPool);
    OA_LOAD_DEVICE(DestroyDescriptorPool, vkDestroyDescriptorPool);
    OA_LOAD_DEVICE(AllocateDescriptorSets, vkAllocateDescriptorSets);
    OA_LOAD_DEVICE(UpdateDescriptorSets, vkUpdateDescriptorSets);
    OA_LOAD_DEVICE(CreateShaderModule, vkCreateShaderModule);
    OA_LOAD_DEVICE(DestroyShaderModule, vkDestroyShaderModule);
    OA_LOAD_DEVICE(CreatePipelineLayout, vkCreatePipelineLayout);
    OA_LOAD_DEVICE(DestroyPipelineLayout, vkDestroyPipelineLayout);
    OA_LOAD_DEVICE(CreateComputePipelines, vkCreateComputePipelines);
    OA_LOAD_DEVICE(DestroyPipeline, vkDestroyPipeline);
    OA_LOAD_DEVICE(CreateCommandPool, vkCreateCommandPool);
    OA_LOAD_DEVICE(DestroyCommandPool, vkDestroyCommandPool);
    OA_LOAD_DEVICE(AllocateCommandBuffers, vkAllocateCommandBuffers);
    OA_LOAD_DEVICE(BeginCommandBuffer, vkBeginCommandBuffer);
    OA_LOAD_DEVICE(EndCommandBuffer, vkEndCommandBuffer);
    OA_LOAD_DEVICE(CmdPipelineBarrier, vkCmdPipelineBarrier);
    OA_LOAD_DEVICE(CmdBindPipeline, vkCmdBindPipeline);
    OA_LOAD_DEVICE(CmdBindDescriptorSets, vkCmdBindDescriptorSets);
    OA_LOAD_DEVICE(CmdDispatch, vkCmdDispatch);
    OA_LOAD_DEVICE(CreateFence, vkCreateFence);
    OA_LOAD_DEVICE(DestroyFence, vkDestroyFence);
    OA_LOAD_DEVICE(QueueSubmit, vkQueueSubmit);
    OA_LOAD_DEVICE(WaitForFences, vkWaitForFences);
#undef OA_LOAD_DEVICE
    return functions;
}

bool HasExtension(const std::vector<VkExtensionProperties>& extensions, const char* name) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}

struct FeatureCaps {
    VkPhysicalDeviceVulkan11Features Core11{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features Core12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features Core13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures ExtBda{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceDescriptorIndexingFeatures ExtDescriptor{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceTimelineSemaphoreFeatures ExtTimeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceSynchronization2Features ExtSync2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
    VkPhysicalDeviceDynamicRenderingFeatures ExtDynamicRendering{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
};

FeatureCaps QueryFeatureCaps(
    const InstanceDispatch& vk,
    VkPhysicalDevice physicalDevice,
    std::uint32_t deviceApi,
    const std::vector<VkExtensionProperties>& extensions) {
    FeatureCaps caps;

    VkPhysicalDeviceFeatures2 core{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    core.pNext = &caps.Core11;
    caps.Core11.pNext = &caps.Core12;
    caps.Core12.pNext = &caps.Core13;
    vk.GetPhysicalDeviceFeatures2(physicalDevice, &core);

    VkPhysicalDeviceFeatures2 extensionFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    void** tail = &extensionFeatures.pNext;
    const auto append = [&tail](auto& feature) {
        *tail = &feature;
        tail = &feature.pNext;
    };

    if (deviceApi < VK_API_VERSION_1_2) {
        if (HasExtension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
            append(caps.ExtBda);
        }
        if (HasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            append(caps.ExtDescriptor);
        }
        if (HasExtension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
            append(caps.ExtTimeline);
        }
    }
    if (deviceApi < VK_API_VERSION_1_3) {
        if (HasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
            append(caps.ExtSync2);
        }
        if (HasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
            append(caps.ExtDynamicRendering);
        }
    }
    if (extensionFeatures.pNext != nullptr) {
        vk.GetPhysicalDeviceFeatures2(physicalDevice, &extensionFeatures);
    }
    return caps;
}

std::uint32_t FindHostMemoryType(
    const VkPhysicalDeviceMemoryProperties& memory,
    std::uint32_t allowedTypes,
    bool& coherent) {
    std::uint32_t fallback = UINT32_MAX;
    for (std::uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
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
        Fail("No host-visible Vulkan memory type for probe buffer");
    }
    return fallback;
}

std::string RunComputeDispatch(
    const InstanceDispatch& instanceFunctions,
    VkPhysicalDevice physicalDevice,
    std::uint32_t computeQueueFamily) {
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
    Check(instanceFunctions.CreateDevice(physicalDevice, &deviceInfo, nullptr, &device),
          "vkCreateDevice(minimal probe)");
    DeviceDispatch vk = LoadDeviceDispatch(instanceFunctions.GetDeviceProcAddr, device);

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
        constexpr VkDeviceSize bufferSize = ProbeElementCount * sizeof(std::uint32_t);
        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        Check(vk.CreateBuffer(device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements requirements{};
        vk.GetBufferMemoryRequirements(device, buffer, &requirements);
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        instanceFunctions.GetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
        bool coherent = false;
        const std::uint32_t memoryType = FindHostMemoryType(
            memoryProperties, requirements.memoryTypeBits, coherent);

        VkMemoryAllocateInfo allocationInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size,
            .memoryTypeIndex = memoryType,
        };
        Check(vk.AllocateMemory(device, &allocationInfo, nullptr, &memory), "vkAllocateMemory");
        Check(vk.BindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");
        Check(vk.MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");

        auto* values = static_cast<std::uint32_t*>(mapped);
        for (std::uint32_t index = 0; index < ProbeElementCount; ++index) {
            values[index] = index;
        }
        if (!coherent) {
            VkMappedMemoryRange range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = memory,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            Check(vk.FlushMappedMemoryRanges(device, 1, &range), "vkFlushMappedMemoryRanges");
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
        Check(vk.CreateDescriptorSetLayout(
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
        Check(vk.CreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool),
              "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo setInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptorPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptorLayout,
        };
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        Check(vk.AllocateDescriptorSets(device, &setInfo, &descriptorSet),
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
        vk.UpdateDescriptorSets(device, 1, &write, 0, nullptr);

        VkShaderModuleCreateInfo shaderInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = OaProbeShaderSpirvSize,
            .pCode = OaProbeShaderSpirv,
        };
        Check(vk.CreateShaderModule(device, &shaderInfo, nullptr, &shader),
              "vkCreateShaderModule");
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptorLayout,
        };
        Check(vk.CreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout),
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
        Check(vk.CreateComputePipelines(
                  device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
              "vkCreateComputePipelines");

        VkCommandPoolCreateInfo commandPoolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = computeQueueFamily,
        };
        Check(vk.CreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool),
              "vkCreateCommandPool");
        VkCommandBufferAllocateInfo commandBufferInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        Check(vk.AllocateCommandBuffers(device, &commandBufferInfo, &commandBuffer),
              "vkAllocateCommandBuffers");
        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        Check(vk.BeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

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
        vk.CmdPipelineBarrier(
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
        vk.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vk.CmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr);
        vk.CmdDispatch(commandBuffer, 1, 1, 1);

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
        vk.CmdPipelineBarrier(
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
        Check(vk.EndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

        VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        Check(vk.CreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");
        VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };
        VkQueue queue = VK_NULL_HANDLE;
        vk.GetDeviceQueue(device, computeQueueFamily, 0, &queue);
        Check(vk.QueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");
        Check(vk.WaitForFences(device, 1, &fence, VK_TRUE, 10'000'000'000ULL),
              "vkWaitForFences");

        if (!coherent) {
            VkMappedMemoryRange range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .memory = memory,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            Check(vk.InvalidateMappedMemoryRanges(device, 1, &range),
                  "vkInvalidateMappedMemoryRanges");
        }

        for (std::uint32_t index = 0; index < ProbeElementCount; ++index) {
            const std::uint32_t expected = index * 3u + 7u;
            if (values[index] != expected) {
                Fail("Compute result mismatch at index " + std::to_string(index)
                     + ": expected " + std::to_string(expected)
                     + ", got " + std::to_string(values[index]));
            }
        }

        vk.DestroyFence(device, fence, nullptr);
        vk.DestroyCommandPool(device, commandPool, nullptr);
        vk.DestroyPipeline(device, pipeline, nullptr);
        vk.DestroyPipelineLayout(device, pipelineLayout, nullptr);
        vk.DestroyShaderModule(device, shader, nullptr);
        vk.DestroyDescriptorPool(device, descriptorPool, nullptr);
        vk.DestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        vk.UnmapMemory(device, memory);
        vk.FreeMemory(device, memory, nullptr);
        vk.DestroyBuffer(device, buffer, nullptr);
        vk.DestroyDevice(device, nullptr);
        return coherent ? "PASS (64 values, coherent host memory)"
                        : "PASS (64 values, explicit cache management)";
    } catch (...) {
        if (fence != VK_NULL_HANDLE) vk.DestroyFence(device, fence, nullptr);
        if (commandPool != VK_NULL_HANDLE) vk.DestroyCommandPool(device, commandPool, nullptr);
        if (pipeline != VK_NULL_HANDLE) vk.DestroyPipeline(device, pipeline, nullptr);
        if (pipelineLayout != VK_NULL_HANDLE) vk.DestroyPipelineLayout(device, pipelineLayout, nullptr);
        if (shader != VK_NULL_HANDLE) vk.DestroyShaderModule(device, shader, nullptr);
        if (descriptorPool != VK_NULL_HANDLE) vk.DestroyDescriptorPool(device, descriptorPool, nullptr);
        if (descriptorLayout != VK_NULL_HANDLE) {
            vk.DestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        }
        if (mapped != nullptr) vk.UnmapMemory(device, memory);
        if (memory != VK_NULL_HANDLE) vk.FreeMemory(device, memory, nullptr);
        if (buffer != VK_NULL_HANDLE) vk.DestroyBuffer(device, buffer, nullptr);
        vk.DestroyDevice(device, nullptr);
        throw;
    }
}

std::string BuildReport(
    const std::string& source,
    const std::string& driverDirectory,
    const std::string& nativeLibraryDirectory,
    const std::string& cacheDirectory) {
    VulkanLibrary library = OpenVulkanLibrary(
        source, driverDirectory, nativeLibraryDirectory, cacheDirectory);

    InstanceDispatch vk;
    vk.GetInstanceProcAddr = LoadExport<PFN_vkGetInstanceProcAddr>(
        library.Handle, "vkGetInstanceProcAddr");
    vk.EnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vk.GetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    vk.CreateInstance = LoadGlobal<PFN_vkCreateInstance>(
        vk.GetInstanceProcAddr, "vkCreateInstance");

    std::uint32_t loaderApi = VK_API_VERSION_1_0;
    if (vk.EnumerateInstanceVersion != nullptr) {
        Check(vk.EnumerateInstanceVersion(&loaderApi), "vkEnumerateInstanceVersion");
    }
    const std::uint32_t requestedApi = std::min(loaderApi, VK_API_VERSION_1_3);
    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "OA Mobile Vulkan Probe",
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
    Check(vk.CreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");

    try {
#define OA_LOAD_INSTANCE(member, name) \
        vk.member = LoadInstance<PFN_##name>(vk.GetInstanceProcAddr, instance, #name)
        OA_LOAD_INSTANCE(DestroyInstance, vkDestroyInstance);
        OA_LOAD_INSTANCE(EnumeratePhysicalDevices, vkEnumeratePhysicalDevices);
        OA_LOAD_INSTANCE(GetPhysicalDeviceProperties, vkGetPhysicalDeviceProperties);
        OA_LOAD_INSTANCE(GetPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties2);
        OA_LOAD_INSTANCE(GetPhysicalDeviceFeatures2, vkGetPhysicalDeviceFeatures2);
        OA_LOAD_INSTANCE(EnumerateDeviceExtensionProperties, vkEnumerateDeviceExtensionProperties);
        OA_LOAD_INSTANCE(GetPhysicalDeviceQueueFamilyProperties, vkGetPhysicalDeviceQueueFamilyProperties);
        OA_LOAD_INSTANCE(GetPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties);
        OA_LOAD_INSTANCE(CreateDevice, vkCreateDevice);
        OA_LOAD_INSTANCE(GetDeviceProcAddr, vkGetDeviceProcAddr);
#undef OA_LOAD_INSTANCE

        std::uint32_t physicalDeviceCount = 0;
        Check(vk.EnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr),
              "vkEnumeratePhysicalDevices(count)");
        if (physicalDeviceCount == 0) {
            Fail("Vulkan loader returned zero physical devices");
        }
        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        Check(vk.EnumeratePhysicalDevices(
                  instance, &physicalDeviceCount, physicalDevices.data()),
              "vkEnumeratePhysicalDevices(data)");
        VkPhysicalDevice physicalDevice = physicalDevices.front();

        VkPhysicalDeviceProperties properties{};
        vk.GetPhysicalDeviceProperties(physicalDevice, &properties);

        std::uint32_t extensionCount = 0;
        Check(vk.EnumerateDeviceExtensionProperties(
                  physicalDevice, nullptr, &extensionCount, nullptr),
              "vkEnumerateDeviceExtensionProperties(count)");
        std::vector<VkExtensionProperties> extensions(extensionCount);
        Check(vk.EnumerateDeviceExtensionProperties(
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
            || HasExtension(extensions, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)) {
            *propertyTail = &driver;
            propertyTail = &driver.pNext;
        }
        if (properties.apiVersion >= VK_API_VERSION_1_2
            || HasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            *propertyTail = &descriptorLimits;
            propertyTail = &descriptorLimits.pNext;
        }
        if (HasExtension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            *propertyTail = &memoryBudget;
        }
        vk.GetPhysicalDeviceProperties2(physicalDevice, &properties2);

        FeatureCaps features = QueryFeatureCaps(
            vk, physicalDevice, properties.apiVersion, extensions);
        const bool core12 = properties.apiVersion >= VK_API_VERSION_1_2;
        const bool core13 = properties.apiVersion >= VK_API_VERSION_1_3;
        const bool bufferDeviceAddress = core12
            ? features.Core12.bufferDeviceAddress == VK_TRUE
            : features.ExtBda.bufferDeviceAddress == VK_TRUE;
        const bool descriptorIndexing = core12
            ? features.Core12.descriptorIndexing == VK_TRUE
            : HasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        const auto descriptorValue = [core12](VkBool32 core, VkBool32 extension) {
            return core12 ? core : extension;
        };
        const bool timelineSemaphore = core12
            ? features.Core12.timelineSemaphore == VK_TRUE
            : features.ExtTimeline.timelineSemaphore == VK_TRUE;
        const bool synchronization2 = core13
            ? features.Core13.synchronization2 == VK_TRUE
            : features.ExtSync2.synchronization2 == VK_TRUE;
        const bool dynamicRendering = core13
            ? features.Core13.dynamicRendering == VK_TRUE
            : features.ExtDynamicRendering.dynamicRendering == VK_TRUE;

        struct Requirement {
            const char* Name;
            bool Supported;
        };
        const std::array requirements{
            Requirement{"bufferDeviceAddress", bufferDeviceAddress},
            Requirement{"descriptorIndexing", descriptorIndexing},
            Requirement{"runtimeDescriptorArray", descriptorValue(
                features.Core12.runtimeDescriptorArray,
                features.ExtDescriptor.runtimeDescriptorArray) == VK_TRUE},
            Requirement{"descriptorBindingPartiallyBound", descriptorValue(
                features.Core12.descriptorBindingPartiallyBound,
                features.ExtDescriptor.descriptorBindingPartiallyBound) == VK_TRUE},
            Requirement{"descriptorBindingVariableDescriptorCount", descriptorValue(
                features.Core12.descriptorBindingVariableDescriptorCount,
                features.ExtDescriptor.descriptorBindingVariableDescriptorCount) == VK_TRUE},
            Requirement{"shaderSampledImageArrayNonUniformIndexing", descriptorValue(
                features.Core12.shaderSampledImageArrayNonUniformIndexing,
                features.ExtDescriptor.shaderSampledImageArrayNonUniformIndexing) == VK_TRUE},
            Requirement{"shaderStorageBufferArrayNonUniformIndexing", descriptorValue(
                features.Core12.shaderStorageBufferArrayNonUniformIndexing,
                features.ExtDescriptor.shaderStorageBufferArrayNonUniformIndexing) == VK_TRUE},
            Requirement{"descriptorBindingStorageBufferUpdateAfterBind", descriptorValue(
                features.Core12.descriptorBindingStorageBufferUpdateAfterBind,
                features.ExtDescriptor.descriptorBindingStorageBufferUpdateAfterBind) == VK_TRUE},
            Requirement{"descriptorBindingStorageImageUpdateAfterBind", descriptorValue(
                features.Core12.descriptorBindingStorageImageUpdateAfterBind,
                features.ExtDescriptor.descriptorBindingStorageImageUpdateAfterBind) == VK_TRUE},
            Requirement{"descriptorBindingSampledImageUpdateAfterBind", descriptorValue(
                features.Core12.descriptorBindingSampledImageUpdateAfterBind,
                features.ExtDescriptor.descriptorBindingSampledImageUpdateAfterBind) == VK_TRUE},
            Requirement{"descriptorBindingUpdateUnusedWhilePending", descriptorValue(
                features.Core12.descriptorBindingUpdateUnusedWhilePending,
                features.ExtDescriptor.descriptorBindingUpdateUnusedWhilePending) == VK_TRUE},
            Requirement{"timelineSemaphore", timelineSemaphore},
            Requirement{"synchronization2", synchronization2},
            Requirement{"dynamicRendering (current OA Core)", dynamicRendering},
        };
        const bool profilePass = std::all_of(
            requirements.begin(),
            requirements.end(),
            [](const Requirement& requirement) { return requirement.Supported; });

        std::uint32_t queueFamilyCount = 0;
        vk.GetPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vk.GetPhysicalDeviceQueueFamilyProperties(
            physicalDevice, &queueFamilyCount, queueFamilies.data());
        std::uint32_t computeQueueFamily = UINT32_MAX;
        for (std::uint32_t index = 0; index < queueFamilyCount; ++index) {
            if ((queueFamilies[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                computeQueueFamily = index;
                break;
            }
        }
        if (computeQueueFamily == UINT32_MAX) {
            Fail("No Vulkan compute queue family");
        }

        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vk.GetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        std::string dispatchResult;
        try {
            dispatchResult = RunComputeDispatch(vk, physicalDevice, computeQueueFamily);
        } catch (const std::exception& error) {
            dispatchResult = std::string("FAIL (") + error.what() + ')';
        }

        std::ostringstream report;
        report << "OA MOBILE LAB / VULKAN PROBE\n"
               << "================================\n"
               << "Driver source: " << (source == "turnip" ? "Bundled Turnip" : "System Adreno") << '\n'
               << "Loader API:    " << VersionString(loaderApi) << '\n'
               << "Instance API:  " << VersionString(requestedApi) << '\n'
               << "Device:        " << properties.deviceName << '\n'
               << "Device API:    " << VersionString(properties.apiVersion) << '\n'
               << "Vendor/Device: 0x" << std::hex << properties.vendorID
               << " / 0x" << properties.deviceID << std::dec << '\n'
               << "Driver ID:     " << driver.driverID << '\n'
               << "Driver name:   " << (driver.driverName[0] == '\0' ? "(not reported)" : driver.driverName) << '\n'
               << "Driver info:   " << (driver.driverInfo[0] == '\0' ? "(not reported)" : driver.driverInfo) << '\n'
               << "Conformance:   "
               << static_cast<unsigned>(driver.conformanceVersion.major) << '.'
               << static_cast<unsigned>(driver.conformanceVersion.minor) << '.'
               << static_cast<unsigned>(driver.conformanceVersion.subminor) << '.'
               << static_cast<unsigned>(driver.conformanceVersion.patch) << '\n'
               << "Extensions:    " << extensions.size() << '\n'
               << "Queue family:  " << computeQueueFamily
               << " (count=" << queueFamilies[computeQueueFamily].queueCount
               << ", timestampBits=" << queueFamilies[computeQueueFamily].timestampValidBits << ")\n"
               << "Subgroup:      size=" << subgroup.subgroupSize
               << ", ops=0x" << std::hex << subgroup.supportedOperations << std::dec << "\n\n";

        report << "OA ModernCompute contract\n"
               << "-------------------------\n"
               << "ModernCompute: " << (profilePass ? "PASS" : "FAIL") << '\n'
               << "Core 1.3 ABI:   " << (core13 ? "yes" : "no — extension adapters required") << '\n';
        for (const Requirement& requirement : requirements) {
            report << (requirement.Supported ? "[+] " : "[-] ") << requirement.Name << '\n';
        }

        report << "\nFeature detail\n"
               << "--------------\n"
               << "16-bit storage:       " << YesNo(features.Core11.storageBuffer16BitAccess) << '\n'
               << "shaderFloat16:        " << YesNo(features.Core12.shaderFloat16) << '\n'
               << "shaderInt8:           " << YesNo(features.Core12.shaderInt8) << '\n'
               << "8-bit storage:        " << YesNo(features.Core12.storageBuffer8BitAccess) << '\n'
               << "BDA provider:         " << (core12 ? "core 1.2" :
                    HasExtension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)
                        ? "VK_KHR_buffer_device_address" : "missing") << '\n'
               << "Descriptor provider:  " << (core12 ? "core 1.2" :
                    HasExtension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)
                        ? "VK_EXT_descriptor_indexing" : "missing") << '\n'
               << "Timeline provider:    " << (core12 ? "core 1.2" :
                    HasExtension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)
                        ? "VK_KHR_timeline_semaphore" : "missing") << '\n'
               << "Sync2 provider:       " << (core13 ? "core 1.3" :
                    HasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)
                        ? "VK_KHR_synchronization2" : "missing") << '\n'
               << "Dynamic provider:     " << (core13 ? "core 1.3" :
                    HasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)
                        ? "VK_KHR_dynamic_rendering" : "missing") << '\n';

        report << "\nDescriptor limits\n"
               << "-----------------\n"
               << "UAB storage buffers/stage: "
               << descriptorLimits.maxPerStageDescriptorUpdateAfterBindStorageBuffers << '\n'
               << "UAB sampled images/stage:  "
               << descriptorLimits.maxPerStageDescriptorUpdateAfterBindSampledImages << '\n'
               << "max storage buffer range:  "
               << BytesString(properties.limits.maxStorageBufferRange) << '\n';

        report << "\nMemory heaps\n"
               << "------------\n";
        for (std::uint32_t heap = 0; heap < memoryProperties.memoryHeapCount; ++heap) {
            report << "heap " << heap << ": " << BytesString(memoryProperties.memoryHeaps[heap].size);
            if ((memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
                report << " device-local";
            }
            if (HasExtension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
                report << ", budget=" << BytesString(memoryBudget.heapBudget[heap])
                       << ", usage=" << BytesString(memoryBudget.heapUsage[heap]);
            }
            report << '\n';
        }

        report << "\nExecution proof\n"
               << "---------------\n"
               << "Shader:   values[i] = values[i] * 3 + 7\n"
               << "Dispatch: " << dispatchResult << '\n';

        LogInfo("Vulkan probe complete: source=" + source
                + ", device=" + std::string(properties.deviceName)
                + ", API=" + VersionString(properties.apiVersion)
                + ", ModernCompute=" + (profilePass ? "PASS" : "FAIL")
                + ", Dispatch=" + dispatchResult);

        vk.DestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
        return report.str();
    } catch (...) {
        if (instance != VK_NULL_HANDLE && vk.DestroyInstance != nullptr) {
            vk.DestroyInstance(instance, nullptr);
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
    const std::string source = JavaString(env, driverSource).Get();
    const std::string drivers = JavaString(env, driverDirectory).Get();
    const std::string nativeLibraries = JavaString(env, nativeLibraryDirectory).Get();
    const std::string cache = JavaString(env, cacheDirectory).Get();

    try {
        const std::string report = BuildReport(source, drivers, nativeLibraries, cache);
        return env->NewStringUTF(report.c_str());
    } catch (const std::exception& error) {
        const std::string report =
            "OA MOBILE LAB / VULKAN PROBE\n"
            "================================\n"
            "Driver source: " + source + "\n"
            "Fatal: " + error.what() + "\n";
        return env->NewStringUTF(report.c_str());
    }
}
