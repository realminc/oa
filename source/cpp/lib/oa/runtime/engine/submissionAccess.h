#pragma once

#include "engineAccess.h"

namespace oa {

// Private white-box seam for the submission-route contract. Runtime stream and
// graph lowering borrow the sole private engine authority; tests use this
// categorical bridge to prove that raw unknown queue handles are rejected
// without installing raw submission on the public engine surface.
class EngineSubmissionAccess {
public:
	[[nodiscard]] static oavk::Stream* acquireStream(oa::Engine& inEngine) {
		return oa::EngineAccess(inEngine).acquireStream();
	}

	static void releaseStream(
		oa::Engine& inEngine,
		oavk::Stream* inStream)
	{
		oa::EngineAccess(inEngine).releaseStream(inStream);
	}

	[[nodiscard]] static oa::Status submit(
		oa::Engine& inEngine,
		void* inQueue,
		void* inSubmitInfo,
		void* inFence)
	{
		return oa::EngineAccess(inEngine).submitToQueue(
			inQueue, inSubmitInfo, inFence);
	}

	[[nodiscard]] static oa::Status submit2(
		oa::Engine& inEngine,
		void* inQueue,
		const void* inSubmitInfo)
	{
		return oa::EngineAccess(inEngine).submitToQueue2(inQueue, inSubmitInfo);
	}
};

} // namespace oa
