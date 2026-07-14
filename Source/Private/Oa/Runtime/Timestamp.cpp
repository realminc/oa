#include <Oa/Runtime/Timestamp.h>
#include <Oa/Runtime/Engine.h>
#include <Oa/Runtime/Device.h>
#include <Oa/Runtime/Stream.h>
#include <Oa/Runtime/OaVk.h>

OaResult<OaVkTimestamp> OaVkTimestamp::Create(OaComputeEngine &InRt, OaU32 InMaxQueries) {
	OaVkTimestamp ts;
	ts.Capacity = InMaxQueries;
	ts.WriteIndex = 0;
	ts.Results.Resize(InMaxQueries, 0);

	auto *dev = static_cast<VkDevice>(InRt.Device.Device);
	auto *physDev = static_cast<VkPhysicalDevice>(InRt.Device.PhysicalDevice);

	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(physDev, &props);
	ts.NanosPerTick = static_cast<OaF64>(props.limits.timestampPeriod);

	if (ts.NanosPerTick <= 0.0) {
		return OaStatus::Error("GPU does not support timestamps (timestampPeriod == 0)");
	}

	VkQueryPoolCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
	ci.queryCount = InMaxQueries;

	VkQueryPool pool = VK_NULL_HANDLE;
	VkResult res = vkCreateQueryPool(dev, &ci, nullptr, &pool);
	if (res not_eq VK_SUCCESS) {
		return OaStatus::Error(OaStatusCode::VulkanError, "vkCreateQueryPool failed");
	}

	ts.Pool = pool;
	return std::move(ts);
}

void OaVkTimestamp::Destroy(const OaVkDevice &InDevice) {
	if (Pool) {
		auto *dev = static_cast<VkDevice>(InDevice.Device);
		vkDestroyQueryPool(dev, static_cast<VkQueryPool>(Pool), nullptr);
		Pool = nullptr;
	}
}

void OaVkTimestamp::Reset(OaVkStream *InStream) {
	WriteIndex = 0;
	auto *cmd = static_cast<VkCommandBuffer>(InStream->CommandBuffer);
	vkCmdResetQueryPool(cmd, static_cast<VkQueryPool>(Pool), 0, Capacity);
}

void OaVkTimestamp::WriteTimestamp(OaVkStream *InStream) {
	if (WriteIndex >= Capacity) return;
	auto *cmd = static_cast<VkCommandBuffer>(InStream->CommandBuffer);
	vkCmdWriteTimestamp2(cmd,
		VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		static_cast<VkQueryPool>(Pool),
		WriteIndex);
	++WriteIndex;
}

OaStatus OaVkTimestamp::Readback(const OaVkDevice &InDevice) {
	if (WriteIndex == 0) return OaStatus::Ok();
	auto *dev = static_cast<VkDevice>(InDevice.Device);
	VkResult res = vkGetQueryPoolResults(
		dev,
		static_cast<VkQueryPool>(Pool),
		0, WriteIndex,
		WriteIndex * sizeof(OaU64),
		Results.Data(),
		sizeof(OaU64),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
	);
	if (res not_eq VK_SUCCESS) {
		return OaStatus::Error("vkGetQueryPoolResults failed");
	}
	return OaStatus::Ok();
}

OaF64 OaVkTimestamp::ElapsedMs(OaU32 InStartIdx, OaU32 InEndIdx) const {
	if (InStartIdx >= WriteIndex or InEndIdx >= WriteIndex or InEndIdx <= InStartIdx) {
		return 0.0;
	}
	OaU64 delta = Results[InEndIdx] - Results[InStartIdx];
	return static_cast<OaF64>(delta) * NanosPerTick / 1e6;
}

OaF64 OaVkTimestamp::ElapsedNs(OaU32 InStartIdx, OaU32 InEndIdx) const {
	if (InStartIdx >= WriteIndex or InEndIdx >= WriteIndex or InEndIdx <= InStartIdx) {
		return 0.0;
	}
	OaU64 delta = Results[InEndIdx] - Results[InStartIdx];
	return static_cast<OaF64>(delta) * NanosPerTick;
}

OaVkTimestamp::OaVkTimestamp(OaVkTimestamp &&InOther) noexcept
	: Pool(InOther.Pool), Capacity(InOther.Capacity), WriteIndex(InOther.WriteIndex),
	  NanosPerTick(InOther.NanosPerTick), Results(std::move(InOther.Results)) {
	InOther.Pool = nullptr;
	InOther.Capacity = 0;
	InOther.WriteIndex = 0;
}

OaVkTimestamp &OaVkTimestamp::operator=(OaVkTimestamp &&InOther) noexcept {
	if (this not_eq &InOther) {
		Pool = InOther.Pool;
		Capacity = InOther.Capacity;
		WriteIndex = InOther.WriteIndex;
		NanosPerTick = InOther.NanosPerTick;
		Results = std::move(InOther.Results);
		InOther.Pool = nullptr;
		InOther.Capacity = 0;
		InOther.WriteIndex = 0;
	}
	return *this;
}
