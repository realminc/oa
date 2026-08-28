#include <oa/runtime/timestamp.h>
#include "engine/deviceAccess.h"
#include <oa/runtime/engine.h>
#include <oa/runtime/device.h>
#include <oa/runtime/stream.h>
#include <vkl/vkl.h>

oa::Result<oavk::Timestamp> oavk::Timestamp::create(oa::Engine &inRt, oa::U32 inMaxQueries) {
	oavk::Timestamp ts;
	ts.capacity = inMaxQueries;
	ts.writeIndex = 0;
	ts.results.resize(inMaxQueries, 0);

	const auto& device = oa::EngineDeviceAccess::get(inRt);
	auto *dev = static_cast<VkDevice>(device.device);
	ts.nanosPerTick = device.info.hardware.timestampPeriodNanoseconds;
	ts.validBits = device.info.hardware.computeTimestampValidBits;

	if (ts.nanosPerTick <= 0.0 or ts.validBits == 0U) {
		return oa::Status::error(oa::StatusCode::Unavailable,
			"compute queue does not support vulkan timestamp queries");
	}

	VkQueryPoolCreateInfo ci{};
	ci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	ci.queryType = VK_QUERY_TYPE_TIMESTAMP;
	ci.queryCount = inMaxQueries;

	VkQueryPool pool = VK_NULL_HANDLE;
	VkResult res = device.deviceDispatch.vkCreateQueryPool(dev, &ci, nullptr, &pool);
	if (res not_eq VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError, "vkCreateQueryPool failed");
	}

	ts.pool = pool;
	ts.owner_ = &inRt;
	return oa::move(ts);
}

oavk::Timestamp::~Timestamp() {
	reset_();
}

void oavk::Timestamp::reset_() noexcept {
	if (pool != nullptr and owner_ != nullptr and owner_->isReady()) {
		destroyDevice_(oa::EngineDeviceAccess::get(*owner_));
	} else {
		pool = nullptr;
		owner_ = nullptr;
	}
}

void oavk::Timestamp::destroyDevice_(const oavk::Device &inDevice) {
	if (pool) {
		auto *dev = static_cast<VkDevice>(inDevice.device);
		inDevice.deviceDispatch.vkDestroyQueryPool(
			dev, static_cast<VkQueryPool>(pool), nullptr);
		pool = nullptr;
	}
	owner_ = nullptr;
}

void oavk::Timestamp::reset(oavk::Stream *inStream) {
	writeIndex = 0;
	auto *cmd = static_cast<VkCommandBuffer>(inStream->commandBuffer);
	inStream->deviceDispatch->vkCmdResetQueryPool(
		cmd, static_cast<VkQueryPool>(pool), 0, capacity);
}

void oavk::Timestamp::writeTimestamp(oavk::Stream *inStream) {
	if (writeIndex >= capacity) return;
	auto *cmd = static_cast<VkCommandBuffer>(inStream->commandBuffer);
	inStream->deviceDispatch->vkCmdWriteTimestamp2(cmd,
		VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
		static_cast<VkQueryPool>(pool),
		writeIndex);
	++writeIndex;
}

oa::Status oavk::Timestamp::readback(const oa::Engine &inEngine) {
	if (owner_ != &inEngine) {
		return oa::Status::invalidArgument(
			"timestamp readback requires the engine that created the query pool");
	}
	return readbackDevice_(oa::EngineDeviceAccess::get(inEngine));
}

oa::Status oavk::Timestamp::readbackDevice_(const oavk::Device &inDevice) {
	if (writeIndex == 0) return oa::Status::ok();
	auto *dev = static_cast<VkDevice>(inDevice.device);
	VkResult res = inDevice.deviceDispatch.vkGetQueryPoolResults(
		dev,
		static_cast<VkQueryPool>(pool),
		0, writeIndex,
		writeIndex * sizeof(oa::U64),
		results.data(),
		sizeof(oa::U64),
		VK_QUERY_RESULT_64_BIT
	);
	if (res == VK_NOT_READY) {
		return oa::Status::error(oa::StatusCode::FailedPrecondition,
			"timestamp results are not ready; complete the submission event first");
	}
	if (res not_eq VK_SUCCESS) {
		return oa::Status::error(oa::StatusCode::VulkanError,
			"vkGetQueryPoolResults failed");
	}
	return oa::Status::ok();
}

oa::F64 oavk::Timestamp::elapsedMs(oa::U32 inStartIdx, oa::U32 inEndIdx) const {
	if (inStartIdx >= writeIndex or inEndIdx >= writeIndex or inEndIdx <= inStartIdx) {
		return 0.0;
	}
	return elapsedNs(inStartIdx, inEndIdx) / 1e6;
}

oa::F64 oavk::Timestamp::elapsedNs(oa::U32 inStartIdx, oa::U32 inEndIdx) const {
	if (inStartIdx >= writeIndex or inEndIdx >= writeIndex or inEndIdx <= inStartIdx) {
		return 0.0;
	}
	if (validBits == 0U or validBits > 64U) return 0.0;
	oa::U64 delta = results[inEndIdx] - results[inStartIdx];
	if (validBits < 64U) {
		delta &= (oa::U64{1} << validBits) - 1U;
	}
	return static_cast<oa::F64>(delta) * nanosPerTick;
}

oavk::Timestamp::Timestamp(oavk::Timestamp &&inOther) noexcept
	: pool(inOther.pool), capacity(inOther.capacity), writeIndex(inOther.writeIndex),
	  nanosPerTick(inOther.nanosPerTick), validBits(inOther.validBits),
	  results(oa::move(inOther.results)), owner_(inOther.owner_) {
	inOther.pool = nullptr;
	inOther.owner_ = nullptr;
	inOther.capacity = 0;
	inOther.writeIndex = 0;
	inOther.validBits = 0;
}

oavk::Timestamp &oavk::Timestamp::operator=(oavk::Timestamp &&inOther) noexcept {
	if (this not_eq &inOther) {
		reset_();
		pool = inOther.pool;
		capacity = inOther.capacity;
		writeIndex = inOther.writeIndex;
		nanosPerTick = inOther.nanosPerTick;
		validBits = inOther.validBits;
		results = oa::move(inOther.results);
		owner_ = inOther.owner_;
		inOther.pool = nullptr;
		inOther.owner_ = nullptr;
		inOther.capacity = 0;
		inOther.writeIndex = 0;
		inOther.validBits = 0;
	}
	return *this;
}
