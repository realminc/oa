// SPDX: MIT. Derived from Vulkan Memory Allocator, (c) 2017-2026 Advanced Micro Devices, Inc.
// Full MIT copyright/permission notice: source/cpp/thirdparty/vma/vma.h. See NOTICE.md.
// Vulkan dispatch is supplied explicitly by OA's allocator integration. This
// implementation has no loader dependency.
namespace vma::detail {

#ifndef _VMA_PUBLIC_INTERFACE

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAllocator(
	const VmaAllocatorCreateInfo* pCreateInfo,
	VmaAllocator* pAllocator)
{
	VMA_ASSERT(pCreateInfo && pAllocator);
	VMA_ASSERT(pCreateInfo->vulkanApiVersion == 0 ||
		(VK_VERSION_MAJOR(pCreateInfo->vulkanApiVersion) == 1 && VK_VERSION_MINOR(pCreateInfo->vulkanApiVersion) <= 4));
	VMA_DEBUG_LOG("vmaCreateAllocator");
	*pAllocator = Vma_new(pCreateInfo->pAllocationCallbacks, VmaAllocator_T)(pCreateInfo);
	VkResult result = (*pAllocator)->Init(pCreateInfo);
	if(result < 0)
	{
		Vma_delete(pCreateInfo->pAllocationCallbacks, *pAllocator);
		*pAllocator = VK_NULL_HANDLE;
	}
	return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyAllocator(
	VmaAllocator allocator)
{
	if(allocator != VK_NULL_HANDLE)
	{
		VMA_DEBUG_LOG("vmaDestroyAllocator");
		VkAllocationCallbacks allocationCallbacks = allocator->m_AllocationCallbacks; // Have to copy the callbacks when destroying.
		Vma_delete(&allocationCallbacks, allocator);
	}
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocatorInfo(VmaAllocator allocator, VmaAllocatorInfo* pAllocatorInfo)
{
	VMA_ASSERT(allocator && pAllocatorInfo);
	pAllocatorInfo->instance = allocator->m_hInstance;
	pAllocatorInfo->physicalDevice = allocator->GetPhysicalDevice();
	pAllocatorInfo->device = allocator->m_hDevice;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPhysicalDeviceProperties(
	VmaAllocator allocator,
	const VkPhysicalDeviceProperties **ppPhysicalDeviceProperties)
{
	VMA_ASSERT(allocator && ppPhysicalDeviceProperties);
	*ppPhysicalDeviceProperties = &allocator->m_PhysicalDeviceProperties;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryProperties(
	VmaAllocator allocator,
	const VkPhysicalDeviceMemoryProperties** ppPhysicalDeviceMemoryProperties)
{
	VMA_ASSERT(allocator && ppPhysicalDeviceMemoryProperties);
	*ppPhysicalDeviceMemoryProperties = &allocator->m_MemProps;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetMemoryTypeProperties(
	VmaAllocator allocator,
	uint32_t memoryTypeIndex,
	VkMemoryPropertyFlags* pFlags)
{
	VMA_ASSERT(allocator && pFlags);
	VMA_ASSERT(memoryTypeIndex < allocator->GetMemoryTypeCount());
	*pFlags = allocator->m_MemProps.memoryTypes[memoryTypeIndex].propertyFlags;
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetCurrentFrameIndex(
	VmaAllocator allocator,
	uint32_t frameIndex)
{
	VMA_ASSERT(allocator);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->SetCurrentFrameIndex(frameIndex);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateStatistics(
	VmaAllocator allocator,
	VmaTotalStatistics* pStats)
{
	VMA_ASSERT(allocator && pStats);
	VMA_DEBUG_GLOBAL_MUTEX_LOCK
	allocator->CalculateStatistics(pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetHeapBudgets(
	VmaAllocator allocator,
	VmaBudget* pBudgets)
{
	VMA_ASSERT(allocator && pBudgets);
	VMA_DEBUG_GLOBAL_MUTEX_LOCK
	allocator->GetHeapBudgets(pBudgets, 0, allocator->GetMemoryHeapCount());
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildStatsString(
	VmaAllocator allocator,
	char** ppStatsString,
	VkBool32 detailedMap)
{
	VMA_ASSERT(allocator && ppStatsString);
	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VmaStringBuilder sb(allocator->GetAllocationCallbacks());
	{
		VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
		allocator->GetHeapBudgets(budgets, 0, allocator->GetMemoryHeapCount());

		VmaTotalStatistics stats;
		allocator->CalculateStatistics(&stats);

		VmaJsonWriter json(allocator->GetAllocationCallbacks(), sb);
		json.BeginObject();
		{
			json.WriteString("General");
			json.BeginObject();
			{
				const VkPhysicalDeviceProperties& deviceProperties = allocator->m_PhysicalDeviceProperties;
				const VkPhysicalDeviceMemoryProperties& memoryProperties = allocator->m_MemProps;

				json.WriteString("API");
				json.WriteString("Vulkan");

				json.WriteString("apiVersion");
				json.BeginString();
				json.ContinueString(VK_VERSION_MAJOR(deviceProperties.apiVersion));
				json.ContinueString(".");
				json.ContinueString(VK_VERSION_MINOR(deviceProperties.apiVersion));
				json.ContinueString(".");
				json.ContinueString(VK_VERSION_PATCH(deviceProperties.apiVersion));
				json.EndString();

				json.WriteString("GPU");
				json.WriteString(deviceProperties.deviceName);
				json.WriteString("deviceType");
				json.WriteNumber(static_cast<uint32_t>(deviceProperties.deviceType));

				json.WriteString("maxMemoryAllocationCount");
				json.WriteNumber(deviceProperties.limits.maxMemoryAllocationCount);
				json.WriteString("bufferImageGranularity");
				json.WriteNumber(deviceProperties.limits.bufferImageGranularity);
				json.WriteString("nonCoherentAtomSize");
				json.WriteNumber(deviceProperties.limits.nonCoherentAtomSize);

				json.WriteString("memoryHeapCount");
				json.WriteNumber(memoryProperties.memoryHeapCount);
				json.WriteString("memoryTypeCount");
				json.WriteNumber(memoryProperties.memoryTypeCount);
			}
			json.EndObject();
		}
		{
			json.WriteString("Total");
			VmaPrintDetailedStatistics(json, stats.total);
		}
		{
			json.WriteString("MemoryInfo");
			json.BeginObject();
			{
				for (uint32_t heapIndex = 0; heapIndex < allocator->GetMemoryHeapCount(); ++heapIndex)
				{
					json.BeginString("Heap ");
					json.ContinueString(heapIndex);
					json.EndString();
					json.BeginObject();
					{
						const VkMemoryHeap& heapInfo = allocator->m_MemProps.memoryHeaps[heapIndex];
						json.WriteString("Flags");
						json.BeginArray(true);
						{
							if (heapInfo.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
								json.WriteString("DEVICE_LOCAL");
						#if VMA_VULKAN_VERSION >= 1001000
							if (heapInfo.flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT)
								json.WriteString("MULTI_INSTANCE");
						#endif

							VkMemoryHeapFlags flags = heapInfo.flags &
								~(VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
						#if VMA_VULKAN_VERSION >= 1001000
									| VK_MEMORY_HEAP_MULTI_INSTANCE_BIT
						#endif
									);
							if (flags != 0)
								json.WriteNumber(flags);
						}
						json.EndArray();

						json.WriteString("Size");
						json.WriteNumber(heapInfo.size);

						json.WriteString("Budget");
						json.BeginObject();
						{
							json.WriteString("BudgetBytes");
							json.WriteNumber(budgets[heapIndex].budget);
							json.WriteString("UsageBytes");
							json.WriteNumber(budgets[heapIndex].usage);
						}
						json.EndObject();

						json.WriteString("Stats");
						VmaPrintDetailedStatistics(json, stats.memoryHeap[heapIndex]);

						json.WriteString("MemoryPools");
						json.BeginObject();
						{
							for (uint32_t typeIndex = 0; typeIndex < allocator->GetMemoryTypeCount(); ++typeIndex)
							{
								if (allocator->MemoryTypeIndexToHeapIndex(typeIndex) == heapIndex)
								{
									json.BeginString("Type ");
									json.ContinueString(typeIndex);
									json.EndString();
									json.BeginObject();
									{
										json.WriteString("Flags");
										json.BeginArray(true);
										{
											VkMemoryPropertyFlags flags = allocator->m_MemProps.memoryTypes[typeIndex].propertyFlags;
											if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
												json.WriteString("DEVICE_LOCAL");
											if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
												json.WriteString("HOST_VISIBLE");
											if (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
												json.WriteString("HOST_COHERENT");
											if (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
												json.WriteString("HOST_CACHED");
											if (flags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
												json.WriteString("LAZILY_ALLOCATED");
										#if VMA_VULKAN_VERSION >= 1001000
											if (flags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
												json.WriteString("PROTECTED");
										#endif
										#if VK_AMD_device_coherent_memory
											if (flags & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY)
												json.WriteString("DEVICE_COHERENT_AMD");
											if (flags & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY)
												json.WriteString("DEVICE_UNCACHED_AMD");
										#endif

											flags &= ~(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
										#if VMA_VULKAN_VERSION >= 1001000
												| VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT
										#endif
										#if VK_AMD_device_coherent_memory
												| VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD_COPY
												| VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD_COPY
										#endif
												| VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
												| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
												| VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
											if (flags != 0)
												json.WriteNumber(flags);
										}
										json.EndArray();

										json.WriteString("Stats");
										VmaPrintDetailedStatistics(json, stats.memoryType[typeIndex]);
									}
									json.EndObject();
								}
							}

						}
						json.EndObject();
					}
					json.EndObject();
				}
			}
			json.EndObject();
		}

		if (detailedMap == VK_TRUE)
			allocator->PrintDetailedMap(json);

		json.EndObject();
	}

	*ppStatsString = VmaCreateStringCopy(allocator->GetAllocationCallbacks(), sb.GetData(), sb.GetLength());
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeStatsString(
	VmaAllocator allocator,
	char* pStatsString)
{
	if(pStatsString != VMA_NULL)
	{
		VMA_ASSERT(allocator);
		VmaFreeString(allocator->GetAllocationCallbacks(), pStatsString);
	}
}

#endif // VMA_STATS_STRING_ENABLED

/*
This function is not protected by any mutex because it just reads immutable data.
*/
VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndex(
	VmaAllocator allocator,
	uint32_t memoryTypeBits,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	uint32_t* pMemoryTypeIndex)
{
	VMA_ASSERT(allocator != VK_NULL_HANDLE);
	VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
	VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

	return allocator->FindMemoryTypeIndex(memoryTypeBits, pAllocationCreateInfo, VmaBufferImageUsage::UNKNOWN, pMemoryTypeIndex);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForBufferInfo(
	VmaAllocator allocator,
	const VkBufferCreateInfo* pBufferCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	uint32_t* pMemoryTypeIndex)
{
	VMA_ASSERT(allocator != VK_NULL_HANDLE);
	VMA_ASSERT(pBufferCreateInfo != VMA_NULL);
	VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
	VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

	const VkDevice hDev = allocator->m_hDevice;
	const VmaVulkanFunctions* funcs = &allocator->GetVulkanFunctions();
	VkResult res = VK_SUCCESS;

#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
	if (funcs->vkGetDeviceBufferMemoryRequirements &&
		(allocator->m_UseKhrMaintenance4 || allocator->m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0)))
	{
		// Can query straight from VkBufferCreateInfo :)
		VkDeviceBufferMemoryRequirementsKHR devBufMemReq = {VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS_KHR};
		devBufMemReq.pCreateInfo = pBufferCreateInfo;

		VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
		(*funcs->vkGetDeviceBufferMemoryRequirements)(hDev, &devBufMemReq, &memReq);

		return allocator->FindMemoryTypeIndex(
			memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo,
			VmaBufferImageUsage(*pBufferCreateInfo, allocator->m_UseKhrMaintenance5), pMemoryTypeIndex);
	}
#endif // VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000

	// Must create a dummy buffer to query :(
	VkBuffer hBuffer = VK_NULL_HANDLE;
	res = funcs->vkCreateBuffer(
		hDev, pBufferCreateInfo, allocator->GetAllocationCallbacks(), &hBuffer);
	if(res == VK_SUCCESS)
	{
		VkMemoryRequirements memReq = {};
		funcs->vkGetBufferMemoryRequirements(hDev, hBuffer, &memReq);

		res = allocator->FindMemoryTypeIndex(
			memReq.memoryTypeBits, pAllocationCreateInfo,
			VmaBufferImageUsage(*pBufferCreateInfo, allocator->m_UseKhrMaintenance5), pMemoryTypeIndex);

		funcs->vkDestroyBuffer(
			hDev, hBuffer, allocator->GetAllocationCallbacks());
	}
	return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFindMemoryTypeIndexForImageInfo(
	VmaAllocator allocator,
	const VkImageCreateInfo* pImageCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	uint32_t* pMemoryTypeIndex)
{
	VMA_ASSERT(allocator != VK_NULL_HANDLE);
	VMA_ASSERT(pImageCreateInfo != VMA_NULL);
	VMA_ASSERT(pAllocationCreateInfo != VMA_NULL);
	VMA_ASSERT(pMemoryTypeIndex != VMA_NULL);

	const VkDevice hDev = allocator->m_hDevice;
	const VmaVulkanFunctions* funcs = &allocator->GetVulkanFunctions();
	VkResult res = VK_SUCCESS;

#if VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000
	if(funcs->vkGetDeviceImageMemoryRequirements &&
		(allocator->m_UseKhrMaintenance4 || allocator->m_VulkanApiVersion >= VK_MAKE_VERSION(1, 3, 0)))
	{
		// Can query straight from VkImageCreateInfo :)
		VkDeviceImageMemoryRequirementsKHR devImgMemReq = {VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS_KHR};
		devImgMemReq.pCreateInfo = pImageCreateInfo;
		VMA_ASSERT(pImageCreateInfo->tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT_COPY && (pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
			"Cannot use this VkImageCreateInfo with vmaFindMemoryTypeIndexForImageInfo as I don't know what to pass as VkDeviceImageMemoryRequirements::planeAspect.");

		VkMemoryRequirements2 memReq = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
		(*funcs->vkGetDeviceImageMemoryRequirements)(hDev, &devImgMemReq, &memReq);

		return allocator->FindMemoryTypeIndex(
			memReq.memoryRequirements.memoryTypeBits, pAllocationCreateInfo,
			VmaBufferImageUsage(*pImageCreateInfo), pMemoryTypeIndex);
	}
#endif // VMA_KHR_MAINTENANCE4 || VMA_VULKAN_VERSION >= 1003000

	// Must create a dummy image to query :(
	VkImage hImage = VK_NULL_HANDLE;
	res = funcs->vkCreateImage(
		hDev, pImageCreateInfo, allocator->GetAllocationCallbacks(), &hImage);
	if(res == VK_SUCCESS)
	{
		VkMemoryRequirements memReq = {};
		funcs->vkGetImageMemoryRequirements(hDev, hImage, &memReq);

		res = allocator->FindMemoryTypeIndex(
			memReq.memoryTypeBits, pAllocationCreateInfo,
			VmaBufferImageUsage(*pImageCreateInfo), pMemoryTypeIndex);

		funcs->vkDestroyImage(
			hDev, hImage, allocator->GetAllocationCallbacks());
	}
	return res;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreatePool(
	VmaAllocator allocator,
	const VmaPoolCreateInfo* pCreateInfo,
	VmaPool* pPool)
{
	VMA_ASSERT(allocator && pCreateInfo && pPool);

	VMA_DEBUG_LOG("vmaCreatePool");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->CreatePool(pCreateInfo, pPool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyPool(
	VmaAllocator allocator,
	VmaPool pool)
{
	VMA_ASSERT(allocator);

	if(pool == VK_NULL_HANDLE)
	{
		return;
	}

	VMA_DEBUG_LOG("vmaDestroyPool");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->DestroyPool(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolStatistics(
	VmaAllocator allocator,
	VmaPool pool,
	VmaStatistics* pPoolStats)
{
	VMA_ASSERT(allocator && pool && pPoolStats);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->GetPoolStatistics(pool, pPoolStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculatePoolStatistics(
	VmaAllocator allocator,
	VmaPool pool,
	VmaDetailedStatistics* pPoolStats)
{
	VMA_ASSERT(allocator && pool && pPoolStats);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->CalculatePoolStatistics(pool, pPoolStats);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckPoolCorruption(VmaAllocator allocator, VmaPool pool)
{
	VMA_ASSERT(allocator && pool);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VMA_DEBUG_LOG("vmaCheckPoolCorruption");

	return allocator->CheckPoolCorruption(pool);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetPoolName(
	VmaAllocator allocator,
	VmaPool pool,
	const char** ppName)
{
	VMA_ASSERT(allocator && pool && ppName);

	VMA_DEBUG_LOG("vmaGetPoolName");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	*ppName = pool->GetName();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetPoolName(
	VmaAllocator allocator,
	VmaPool pool,
	const char* pName)
{
	VMA_ASSERT(allocator && pool);

	VMA_DEBUG_LOG("vmaSetPoolName");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	pool->SetName(pName);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemory(
	VmaAllocator allocator,
	const VkMemoryRequirements* pVkMemoryRequirements,
	const VmaAllocationCreateInfo* pCreateInfo,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

	VMA_DEBUG_LOG("vmaAllocateMemory");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkResult result = allocator->AllocateMemory(
		*pVkMemoryRequirements,
		false, // requiresDedicatedAllocation
		false, // prefersDedicatedAllocation
		VK_NULL_HANDLE, // dedicatedBuffer
		VK_NULL_HANDLE, // dedicatedImage
		VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
		VMA_NULL, // pMemoryAllocateNext
		*pCreateInfo,
		VMA_SUBALLOCATION_TYPE_UNKNOWN,
		1, // allocationCount
		pAllocation);

	if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
	{
		allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
	}

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateDedicatedMemory(
	VmaAllocator allocator,
	const VkMemoryRequirements* pVkMemoryRequirements,
	const VmaAllocationCreateInfo* pCreateInfo,
	void* pMemoryAllocateNext,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocation);

	VMA_DEBUG_LOG("vmaAllocateDedicatedMemory");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkResult result = allocator->AllocateMemory(
		*pVkMemoryRequirements,
		true, // requiresDedicatedAllocation
		false, // prefersDedicatedAllocation
		VK_NULL_HANDLE, // dedicatedBuffer
		VK_NULL_HANDLE, // dedicatedImage
		VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
		pMemoryAllocateNext,
		*pCreateInfo,
		VMA_SUBALLOCATION_TYPE_UNKNOWN,
		1, // allocationCount
		pAllocation);

	if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
	{
		allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
	}

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryPages(
	VmaAllocator allocator,
	const VkMemoryRequirements* pVkMemoryRequirements,
	const VmaAllocationCreateInfo* pCreateInfo,
	size_t allocationCount,
	VmaAllocation* pAllocations,
	VmaAllocationInfo* pAllocationInfo)
{
	if(allocationCount == 0)
	{
		return VK_SUCCESS;
	}

	VMA_ASSERT(allocator && pVkMemoryRequirements && pCreateInfo && pAllocations);

	VMA_DEBUG_LOG("vmaAllocateMemoryPages");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkResult result = allocator->AllocateMemory(
		*pVkMemoryRequirements,
		false, // requiresDedicatedAllocation
		false, // prefersDedicatedAllocation
		VK_NULL_HANDLE, // dedicatedBuffer
		VK_NULL_HANDLE, // dedicatedImage
		VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
		VMA_NULL, // pMemoryAllocateNext
		*pCreateInfo,
		VMA_SUBALLOCATION_TYPE_UNKNOWN,
		allocationCount,
		pAllocations);

	if(pAllocationInfo != VMA_NULL && result == VK_SUCCESS)
	{
		for(size_t i = 0; i < allocationCount; ++i)
		{
			allocator->GetAllocationInfo(pAllocations[i], pAllocationInfo + i);
		}
	}

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForBuffer(
	VmaAllocator allocator,
	VkBuffer buffer,
	const VmaAllocationCreateInfo* pCreateInfo,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && buffer != VK_NULL_HANDLE && pCreateInfo && pAllocation);

	VMA_DEBUG_LOG("vmaAllocateMemoryForBuffer");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkMemoryRequirements vkMemReq = {};
	bool requiresDedicatedAllocation = false;
	bool prefersDedicatedAllocation = false;
	allocator->GetBufferMemoryRequirements(buffer, vkMemReq,
		requiresDedicatedAllocation,
		prefersDedicatedAllocation);

	VkResult result = allocator->AllocateMemory(
		vkMemReq,
		requiresDedicatedAllocation,
		prefersDedicatedAllocation,
		buffer, // dedicatedBuffer
		VK_NULL_HANDLE, // dedicatedImage
		VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
		VMA_NULL, // pMemoryAllocateNext
		*pCreateInfo,
		VMA_SUBALLOCATION_TYPE_BUFFER,
		1, // allocationCount
		pAllocation);

	if(pAllocationInfo && result == VK_SUCCESS)
	{
		allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
	}

	return result;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaAllocateMemoryForImage(
	VmaAllocator allocator,
	VkImage image,
	const VmaAllocationCreateInfo* pCreateInfo,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && image != VK_NULL_HANDLE && pCreateInfo && pAllocation);

	VMA_DEBUG_LOG("vmaAllocateMemoryForImage");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	VkMemoryRequirements vkMemReq = {};
	bool requiresDedicatedAllocation = false;
	bool prefersDedicatedAllocation  = false;
	allocator->GetImageMemoryRequirements(image, vkMemReq,
		requiresDedicatedAllocation, prefersDedicatedAllocation);

	VkResult result = allocator->AllocateMemory(
		vkMemReq,
		requiresDedicatedAllocation,
		prefersDedicatedAllocation,
		VK_NULL_HANDLE, // dedicatedBuffer
		image, // dedicatedImage
		VmaBufferImageUsage::UNKNOWN, // dedicatedBufferImageUsage
		VMA_NULL, // pMemoryAllocateNext
		*pCreateInfo,
		VMA_SUBALLOCATION_TYPE_IMAGE_UNKNOWN,
		1, // allocationCount
		pAllocation);

	if(pAllocationInfo && result == VK_SUCCESS)
	{
		allocator->GetAllocationInfo(*pAllocation, pAllocationInfo);
	}

	return result;
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemory(
	VmaAllocator allocator,
	VmaAllocation allocation)
{
	VMA_ASSERT(allocator);

	if(allocation == VK_NULL_HANDLE)
	{
		return;
	}

	VMA_DEBUG_LOG("vmaFreeMemory");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->FreeMemory(
		1, // allocationCount
		&allocation);
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeMemoryPages(
	VmaAllocator allocator,
	size_t allocationCount,
	const VmaAllocation* pAllocations)
{
	if(allocationCount == 0)
	{
		return;
	}

	VMA_ASSERT(allocator);

	VMA_DEBUG_LOG("vmaFreeMemoryPages");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->FreeMemory(allocationCount, pAllocations);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && allocation && pAllocationInfo);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->GetAllocationInfo(allocation, pAllocationInfo);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationInfo2(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VmaAllocationInfo2* pAllocationInfo)
{
	VMA_ASSERT(allocator && allocation && pAllocationInfo);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->GetAllocationInfo2(allocation, pAllocationInfo);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationUserData(
	VmaAllocator allocator,
	VmaAllocation allocation,
	void* pUserData)
{
	VMA_ASSERT(allocator && allocation);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocation->SetUserData(allocator, pUserData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetAllocationName(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation,
	const char* VMA_NULLABLE pName)
{
	allocation->SetName(allocator, pName);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetAllocationMemoryProperties(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation,
	VkMemoryPropertyFlags* VMA_NOT_NULL pFlags)
{
	VMA_ASSERT(allocator && allocation && pFlags);
	const uint32_t memTypeIndex = allocation->GetMemoryTypeIndex();
	*pFlags = allocator->m_MemProps.memoryTypes[memTypeIndex].propertyFlags;
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaMapMemory(
	VmaAllocator allocator,
	VmaAllocation allocation,
	void** ppData)
{
	VMA_ASSERT(allocator && allocation && ppData);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->Map(allocation, ppData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaUnmapMemory(
	VmaAllocator allocator,
	VmaAllocation allocation)
{
	VMA_ASSERT(allocator && allocation);

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	allocator->Unmap(allocation);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocation(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VkDeviceSize offset,
	VkDeviceSize size)
{
	VMA_ASSERT(allocator && allocation);

	VMA_DEBUG_LOG("vmaFlushAllocation");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->FlushOrInvalidateAllocation(allocation, offset, size, VMA_CACHE_FLUSH);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocation(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VkDeviceSize offset,
	VkDeviceSize size)
{
	VMA_ASSERT(allocator && allocation);

	VMA_DEBUG_LOG("vmaInvalidateAllocation");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->FlushOrInvalidateAllocation(allocation, offset, size, VMA_CACHE_INVALIDATE);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaFlushAllocations(
	VmaAllocator allocator,
	uint32_t allocationCount,
	const VmaAllocation* allocations,
	const VkDeviceSize* offsets,
	const VkDeviceSize* sizes)
{
	VMA_ASSERT(allocator);

	if(allocationCount == 0)
	{
		return VK_SUCCESS;
	}

	VMA_ASSERT(allocations);

	VMA_DEBUG_LOG("vmaFlushAllocations");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->FlushOrInvalidateAllocations(allocationCount, allocations, offsets, sizes, VMA_CACHE_FLUSH);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaInvalidateAllocations(
	VmaAllocator allocator,
	uint32_t allocationCount,
	const VmaAllocation* allocations,
	const VkDeviceSize* offsets,
	const VkDeviceSize* sizes)
{
	VMA_ASSERT(allocator);

	if(allocationCount == 0)
	{
		return VK_SUCCESS;
	}

	VMA_ASSERT(allocations);

	VMA_DEBUG_LOG("vmaInvalidateAllocations");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->FlushOrInvalidateAllocations(allocationCount, allocations, offsets, sizes, VMA_CACHE_INVALIDATE);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCopyMemoryToAllocation(
	VmaAllocator allocator,
	const void* pSrcHostPointer,
	VmaAllocation dstAllocation,
	VkDeviceSize dstAllocationLocalOffset,
	VkDeviceSize size)
{
	VMA_ASSERT(allocator && pSrcHostPointer && dstAllocation);

	if(size == 0)
	{
		return VK_SUCCESS;
	}

	VMA_DEBUG_LOG("vmaCopyMemoryToAllocation");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->CopyMemoryToAllocation(pSrcHostPointer, dstAllocation, dstAllocationLocalOffset, size);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCopyAllocationToMemory(
	VmaAllocator allocator,
	VmaAllocation srcAllocation,
	VkDeviceSize srcAllocationLocalOffset,
	void* pDstHostPointer,
	VkDeviceSize size)
{
	VMA_ASSERT(allocator && srcAllocation && pDstHostPointer);

	if(size == 0)
	{
		return VK_SUCCESS;
	}

	VMA_DEBUG_LOG("vmaCopyAllocationToMemory");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->CopyAllocationToMemory(srcAllocation, srcAllocationLocalOffset, pDstHostPointer, size);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCheckCorruption(
	VmaAllocator allocator,
	uint32_t memoryTypeBits)
{
	VMA_ASSERT(allocator);

	VMA_DEBUG_LOG("vmaCheckCorruption");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->CheckCorruption(memoryTypeBits);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentation(
	VmaAllocator allocator,
	const VmaDefragmentationInfo* pInfo,
	VmaDefragmentationContext* pContext)
{
	VMA_ASSERT(allocator && pInfo && pContext);

	VMA_DEBUG_LOG("vmaBeginDefragmentation");

	if (pInfo->pool != VMA_NULL)
	{
		// Check if run on supported algorithms
		if (pInfo->pool->m_BlockVector.GetAlgorithm() & VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT)
			return VK_ERROR_FEATURE_NOT_PRESENT;
	}

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	*pContext = Vma_new(allocator, VmaDefragmentationContext_T)(allocator, *pInfo);
	return VK_SUCCESS;
}

VMA_CALL_PRE void VMA_CALL_POST vmaEndDefragmentation(
	VmaAllocator allocator,
	VmaDefragmentationContext context,
	VmaDefragmentationStats* pStats)
{
	VMA_ASSERT(allocator && context);

	VMA_DEBUG_LOG("vmaEndDefragmentation");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	if (pStats)
		context->GetStats(*pStats);
	Vma_delete(allocator, context);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBeginDefragmentationPass(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaDefragmentationContext VMA_NOT_NULL context,
	VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo)
{
	VMA_ASSERT(context && pPassInfo);

	VMA_DEBUG_LOG("vmaBeginDefragmentationPass");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return context->DefragmentPassBegin(*pPassInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaEndDefragmentationPass(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaDefragmentationContext VMA_NOT_NULL context,
	VmaDefragmentationPassMoveInfo* VMA_NOT_NULL pPassInfo)
{
	VMA_ASSERT(context && pPassInfo);

	VMA_DEBUG_LOG("vmaEndDefragmentationPass");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return context->DefragmentPassEnd(*pPassInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VkBuffer buffer)
{
	VMA_ASSERT(allocator && allocation && buffer);

	VMA_DEBUG_LOG("vmaBindBufferMemory");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->BindBufferMemory(allocation, 0, buffer, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindBufferMemory2(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VkDeviceSize allocationLocalOffset,
	VkBuffer buffer,
	const void* pNext)
{
	VMA_ASSERT(allocator && allocation && buffer);

	VMA_DEBUG_LOG("vmaBindBufferMemory2");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->BindBufferMemory(allocation, allocationLocalOffset, buffer, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VkImage image)
{
	VMA_ASSERT(allocator && allocation && image);

	VMA_DEBUG_LOG("vmaBindImageMemory");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	return allocator->BindImageMemory(allocation, 0, image, VMA_NULL);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaBindImageMemory2(
	VmaAllocator allocator,
	VmaAllocation allocation,
	VkDeviceSize allocationLocalOffset,
	VkImage image,
	const void* pNext)
{
	VMA_ASSERT(allocator && allocation && image);

	VMA_DEBUG_LOG("vmaBindImageMemory2");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

		return allocator->BindImageMemory(allocation, allocationLocalOffset, image, pNext);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBuffer(
	VmaAllocator allocator,
	const VkBufferCreateInfo* pBufferCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	VkBuffer* pBuffer,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);
	VMA_DEBUG_LOG("vmaCreateBuffer");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;

	return allocator->CreateBuffer(pBufferCreateInfo, pAllocationCreateInfo,
		VMA_NULL, // pMemoryAllocateNext
		pBuffer, pAllocation, pAllocationInfo);

}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateBufferWithAlignment(
	VmaAllocator allocator,
	const VkBufferCreateInfo* pBufferCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	VkDeviceSize minAlignment,
	VkBuffer* pBuffer,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && VmaIsPow2(minAlignment) && pBuffer && pAllocation);
	VMA_DEBUG_LOG("vmaCreateBufferWithAlignment");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;

	VmaAllocationCreateInfo allocCreateInfoCopy = *pAllocationCreateInfo;
	allocCreateInfoCopy.minAlignment = VMA_MAX(allocCreateInfoCopy.minAlignment, minAlignment);

	return allocator->CreateBuffer(pBufferCreateInfo, &allocCreateInfoCopy,
		VMA_NULL, // pMemoryAllocateNext
		pBuffer, pAllocation, pAllocationInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateDedicatedBuffer(
	VmaAllocator allocator,
	const VkBufferCreateInfo* pBufferCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	void* pMemoryAllocateNext,
	VkBuffer* pBuffer,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && pBufferCreateInfo && pAllocationCreateInfo && pBuffer && pAllocation);
	VMA_DEBUG_LOG("vmaCreateDedicatedBuffer");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;

	VmaAllocationCreateInfo allocCreateInfoCopy = *pAllocationCreateInfo;
	allocCreateInfoCopy.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

	return allocator->CreateBuffer(pBufferCreateInfo, &allocCreateInfoCopy,
		pMemoryAllocateNext, // pMemoryAllocateNext
		pBuffer, pAllocation, pAllocationInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation,
	const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
	VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer)
{
	return vmaCreateAliasingBuffer2(allocator, allocation, 0, pBufferCreateInfo, pBuffer);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingBuffer2(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation,
	VkDeviceSize allocationLocalOffset,
	const VkBufferCreateInfo* VMA_NOT_NULL pBufferCreateInfo,
	VkBuffer VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pBuffer)
{
	VMA_ASSERT(allocator && pBufferCreateInfo && pBuffer && allocation);
	VMA_ASSERT(allocationLocalOffset + pBufferCreateInfo->size <= allocation->GetSize());

	VMA_DEBUG_LOG("vmaCreateAliasingBuffer2");

	*pBuffer = VK_NULL_HANDLE;

	if (pBufferCreateInfo->size == 0)
	{
		return VK_ERROR_INITIALIZATION_FAILED;
	}
	if ((pBufferCreateInfo->usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_COPY) != 0 &&
		!allocator->m_UseKhrBufferDeviceAddress)
	{
		VMA_ASSERT(0 && "Creating a buffer with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT is not valid if VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT was not used.");
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	// 1. Create VkBuffer.
	VkResult res = (*allocator->GetVulkanFunctions().vkCreateBuffer)(
		allocator->m_hDevice,
		pBufferCreateInfo,
		allocator->GetAllocationCallbacks(),
		pBuffer);
	if (res >= 0)
	{
		// 2. Bind buffer with memory.
		res = allocator->BindBufferMemory(allocation, allocationLocalOffset, *pBuffer, VMA_NULL);
		if (res >= 0)
		{
			return VK_SUCCESS;
		}
		(*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, *pBuffer, allocator->GetAllocationCallbacks());
	}
	return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyBuffer(
	VmaAllocator allocator,
	VkBuffer buffer,
	VmaAllocation allocation)
{
	VMA_ASSERT(allocator);

	if(buffer == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE)
	{
		return;
	}

	VMA_DEBUG_LOG("vmaDestroyBuffer");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	if(buffer != VK_NULL_HANDLE)
	{
		(*allocator->GetVulkanFunctions().vkDestroyBuffer)(allocator->m_hDevice, buffer, allocator->GetAllocationCallbacks());
	}

	if(allocation != VK_NULL_HANDLE)
	{
		allocator->FreeMemory(
			1, // allocationCount
			&allocation);
	}
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateImage(
	VmaAllocator allocator,
	const VkImageCreateInfo* pImageCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	VkImage* pImage,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo)
{
	VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);
	VMA_ASSERT((pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
		"vmaCreateImage() doesn't support disjoint multi-planar images. Please allocate memory for the planes using vmaAllocateMemory() and bind them using vmaBindImageMemory2().");
	VMA_DEBUG_LOG("vmaCreateImage");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;

	return allocator->CreateImage(pImageCreateInfo, pAllocationCreateInfo,
		VMA_NULL, // pMemoryAllocateNext
		pImage, pAllocation, pAllocationInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateDedicatedImage(
	VmaAllocator allocator,
	const VkImageCreateInfo* pImageCreateInfo,
	const VmaAllocationCreateInfo* pAllocationCreateInfo,
	void* pMemoryAllocateNext,
	VkImage* pImage,
	VmaAllocation* pAllocation,
	VmaAllocationInfo* pAllocationInfo
) {
	VMA_ASSERT(allocator && pImageCreateInfo && pAllocationCreateInfo && pImage && pAllocation);
	VMA_ASSERT((pImageCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT_COPY) == 0 &&
		"vmaCreateDedicatedImage() doesn't support disjoint multi-planar images. Please allocate memory for the planes using vmaAllocateMemory() and bind them using vmaBindImageMemory2().");
	VMA_DEBUG_LOG("vmaCreateDedicatedImage");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;

	VmaAllocationCreateInfo allocCreateInfoCopy = *pAllocationCreateInfo;
	allocCreateInfoCopy.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

	return allocator->CreateImage(pImageCreateInfo, &allocCreateInfoCopy,
		pMemoryAllocateNext, // pMemoryAllocateNext
		pImage, pAllocation, pAllocationInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation,
	const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
	VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage) {
	return vmaCreateAliasingImage2(allocator, allocation, 0, pImageCreateInfo, pImage);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateAliasingImage2(
	VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation,
	VkDeviceSize allocationLocalOffset,
	const VkImageCreateInfo* VMA_NOT_NULL pImageCreateInfo,
	VkImage VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pImage
) {
	VMA_ASSERT(allocator && pImageCreateInfo && pImage && allocation);

	*pImage = VK_NULL_HANDLE;

	VMA_DEBUG_LOG("VmaCreateImage2");

	if (pImageCreateInfo->extent.width == 0 ||
		pImageCreateInfo->extent.height == 0 ||
		pImageCreateInfo->extent.depth == 0 ||
		pImageCreateInfo->mipLevels == 0 ||
		pImageCreateInfo->arrayLayers == 0
	) {
		return VK_ERROR_INITIALIZATION_FAILED;
	}

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	// 1. Create VkImage.
	VkResult res = (*allocator->GetVulkanFunctions().vkCreateImage)(
		allocator->m_hDevice,
		pImageCreateInfo,
		allocator->GetAllocationCallbacks(),
		pImage
	);
	if (res >= 0) {
		// 2. Bind image with memory.
		res = allocator->BindImageMemory(allocation, allocationLocalOffset, *pImage, VMA_NULL);
		if (res >= 0) {
			return VK_SUCCESS;
		}
		(*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, *pImage, allocator->GetAllocationCallbacks());
	}
	return res;
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyImage(
	VmaAllocator VMA_NOT_NULL allocator,
	VkImage VMA_NULLABLE_NON_DISPATCHABLE image,
	VmaAllocation VMA_NULLABLE allocation
) {
	VMA_ASSERT(allocator);

	if(image == VK_NULL_HANDLE && allocation == VK_NULL_HANDLE) {
		return;
	}

	VMA_DEBUG_LOG("vmaDestroyImage");

	VMA_DEBUG_GLOBAL_MUTEX_LOCK

	if(image != VK_NULL_HANDLE) {
		(*allocator->GetVulkanFunctions().vkDestroyImage)(allocator->m_hDevice, image, allocator->GetAllocationCallbacks());
	}
	if(allocation != VK_NULL_HANDLE) {
		allocator->FreeMemory(1, &allocation);
	}
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaCreateVirtualBlock(
	const VmaVirtualBlockCreateInfo* VMA_NOT_NULL pCreateInfo,
	VmaVirtualBlock VMA_NULLABLE * VMA_NOT_NULL pVirtualBlock
) {
	VMA_ASSERT(pCreateInfo && pVirtualBlock);
	VMA_ASSERT(pCreateInfo->size > 0);
	VMA_DEBUG_LOG("vmaCreateVirtualBlock");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	*pVirtualBlock = Vma_new(pCreateInfo->pAllocationCallbacks, VmaVirtualBlock_T)(*pCreateInfo);
	return VK_SUCCESS;

	/*
	Code for the future if we ever need a separate Init() method that could fail:

	VkResult res = (*pVirtualBlock)->Init();
	if(res < 0)
	{
		Vma_delete(pCreateInfo->pAllocationCallbacks, *pVirtualBlock);
		*pVirtualBlock = VK_NULL_HANDLE;
	}
	return res;
	*/
}

VMA_CALL_PRE void VMA_CALL_POST vmaDestroyVirtualBlock(VmaVirtualBlock VMA_NULLABLE virtualBlock) {
	if(virtualBlock != VK_NULL_HANDLE) {
		VMA_DEBUG_LOG("vmaDestroyVirtualBlock");
		VMA_DEBUG_GLOBAL_MUTEX_LOCK;
		VkAllocationCallbacks allocationCallbacks = virtualBlock->m_AllocationCallbacks; // Have to copy the callbacks when destroying.
		Vma_delete(&allocationCallbacks, virtualBlock);
	}
}

VMA_CALL_PRE VkBool32 VMA_CALL_POST vmaIsVirtualBlockEmpty(VmaVirtualBlock VMA_NOT_NULL virtualBlock)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
	VMA_DEBUG_LOG("vmaIsVirtualBlockEmpty");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	return virtualBlock->IsEmpty() ? VK_TRUE : VK_FALSE;
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualAllocationInfo(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, VmaVirtualAllocationInfo* VMA_NOT_NULL pVirtualAllocInfo)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pVirtualAllocInfo != VMA_NULL);
	VMA_DEBUG_LOG("vmaGetVirtualAllocationInfo");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	virtualBlock->GetAllocationInfo(allocation, *pVirtualAllocInfo);
}

VMA_CALL_PRE VkResult VMA_CALL_POST vmaVirtualAllocate(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	const VmaVirtualAllocationCreateInfo* VMA_NOT_NULL pCreateInfo, VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE* VMA_NOT_NULL pAllocation,
	VkDeviceSize* VMA_NULLABLE pOffset)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pCreateInfo != VMA_NULL && pAllocation != VMA_NULL);
	VMA_DEBUG_LOG("vmaVirtualAllocate");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	return virtualBlock->Allocate(*pCreateInfo, *pAllocation, pOffset);
}

VMA_CALL_PRE void VMA_CALL_POST vmaVirtualFree(VmaVirtualBlock VMA_NOT_NULL virtualBlock, VmaVirtualAllocation VMA_NULLABLE_NON_DISPATCHABLE allocation)
{
	if(allocation != VK_NULL_HANDLE)
	{
		VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
		VMA_DEBUG_LOG("vmaVirtualFree");
		VMA_DEBUG_GLOBAL_MUTEX_LOCK;
		virtualBlock->Free(allocation);
	}
}

VMA_CALL_PRE void VMA_CALL_POST vmaClearVirtualBlock(VmaVirtualBlock VMA_NOT_NULL virtualBlock)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
	VMA_DEBUG_LOG("vmaClearVirtualBlock");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	virtualBlock->Clear();
}

VMA_CALL_PRE void VMA_CALL_POST vmaSetVirtualAllocationUserData(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	VmaVirtualAllocation VMA_NOT_NULL_NON_DISPATCHABLE allocation, void* VMA_NULLABLE pUserData)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
	VMA_DEBUG_LOG("vmaSetVirtualAllocationUserData");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	virtualBlock->SetAllocationUserData(allocation, pUserData);
}

VMA_CALL_PRE void VMA_CALL_POST vmaGetVirtualBlockStatistics(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	VmaStatistics* VMA_NOT_NULL pStats)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != VMA_NULL);
	VMA_DEBUG_LOG("vmaGetVirtualBlockStatistics");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	virtualBlock->GetStatistics(*pStats);
}

VMA_CALL_PRE void VMA_CALL_POST vmaCalculateVirtualBlockStatistics(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	VmaDetailedStatistics* VMA_NOT_NULL pStats)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && pStats != VMA_NULL);
	VMA_DEBUG_LOG("vmaCalculateVirtualBlockStatistics");
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	virtualBlock->CalculateDetailedStatistics(*pStats);
}

#if VMA_STATS_STRING_ENABLED

VMA_CALL_PRE void VMA_CALL_POST vmaBuildVirtualBlockStatsString(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	char* VMA_NULLABLE * VMA_NOT_NULL ppStatsString, VkBool32 detailedMap)
{
	VMA_ASSERT(virtualBlock != VK_NULL_HANDLE && ppStatsString != VMA_NULL);
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	const VkAllocationCallbacks* allocationCallbacks = virtualBlock->GetAllocationCallbacks();
	VmaStringBuilder sb(allocationCallbacks);
	virtualBlock->BuildStatsString(detailedMap != VK_FALSE, sb);
	*ppStatsString = VmaCreateStringCopy(allocationCallbacks, sb.GetData(), sb.GetLength());
}

VMA_CALL_PRE void VMA_CALL_POST vmaFreeVirtualBlockStatsString(VmaVirtualBlock VMA_NOT_NULL virtualBlock,
	char* VMA_NULLABLE pStatsString)
{
	if(pStatsString != VMA_NULL)
	{
		VMA_ASSERT(virtualBlock != VK_NULL_HANDLE);
		VMA_DEBUG_GLOBAL_MUTEX_LOCK;
		VmaFreeString(virtualBlock->GetAllocationCallbacks(), pStatsString);
	}
}
#if VMA_EXTERNAL_MEMORY_WIN32
VMA_CALL_PRE VkResult VMA_CALL_POST vmaGetMemoryWin32Handle(VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation, HANDLE hTargetProcess, HANDLE* VMA_NOT_NULL pHandle)
{
	VMA_ASSERT(allocator && allocation && pHandle);
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	return allocation->GetWin32Handle(allocator, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, hTargetProcess, pHandle);
}
VMA_CALL_PRE VkResult VMA_CALL_POST vmaGetMemoryWin32Handle2(VmaAllocator VMA_NOT_NULL allocator,
	VmaAllocation VMA_NOT_NULL allocation, VkExternalMemoryHandleTypeFlagBits handleType, HANDLE hTargetProcess, HANDLE* VMA_NOT_NULL pHandle)
{
	VMA_ASSERT(allocator && allocation && pHandle);
	VMA_ASSERT(handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR ||
		handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR ||
		handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT_KHR ||
		handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT_KHR ||
		handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT_KHR ||
		handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT_KHR);
	VMA_DEBUG_GLOBAL_MUTEX_LOCK;
	return allocation->GetWin32Handle(allocator, handleType, hTargetProcess, pHandle);
}
#endif // VMA_EXTERNAL_MEMORY_WIN32
#endif // VMA_STATS_STRING_ENABLED
#endif // _VMA_PUBLIC_INTERFACE

} // namespace vma::detail
