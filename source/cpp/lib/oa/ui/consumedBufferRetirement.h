#pragma once

#include "../runtime/engine/borrowedServiceRetirement.h"

#include <oa/core/types.h>
#include <oa/runtime/allocator.h>
#include <oa/runtime/engine/resourceAccess.h>
#include <oa/runtime/sync.h>

namespace oa {

class ConsumedBufferRetirement {
public:
	static void releaseOrRetire(
		oa::Engine* inEngine,
		oavk::Buffer& inOutBuffer,
		const oa::Event& inConsumer) noexcept
	{
		if (inEngine == nullptr or inOutBuffer.buffer == nullptr) {
			inOutBuffer = {};
			return;
		}

		const bool consumerComplete = not inConsumer.isValid()
			or inConsumer.isComplete();
		if (consumerComplete) {
			oa::EngineResourceAccess::freeBuffer(*inEngine, inOutBuffer);
			inOutBuffer = {};
			return;
		}

		auto retired = oa::makeUnique<Payload>();
		retired->engine = inEngine;
		retired->buffer = inOutBuffer;
		retired->consumer = inConsumer;
		inOutBuffer = {};
		oa::BorrowedServiceRetirement::retire(
			*inEngine,
			retired.release(),
			&complete_,
			&release_);
	}

private:
	struct Payload {
		oa::Engine* engine = nullptr;
		oavk::Buffer buffer;
		oa::Event consumer;
	};

	[[nodiscard]] static oa::Status complete_(void* inPayload)
	{
		auto* retired = static_cast<Payload*>(inPayload);
		if (retired == nullptr or retired->engine == nullptr) {
			return oa::Status::ok();
		}
		OA_RETURN_IF_ERROR(retired->consumer.wait());
		oa::EngineResourceAccess::freeBuffer(
			*retired->engine, retired->buffer);
		retired->buffer = {};
		return oa::Status::ok();
	}

	static void release_(void* inPayload)
	{
		oa::UniquePtr<Payload> retired(static_cast<Payload*>(inPayload));
	}
};

}  // namespace oa
