#pragma once

#include "engineAccess.h"

namespace oa {

class BorrowedServiceRetirement {
public:
	using CompleteFn = oa::Status (*)(void*);
	using ReleaseFn = void (*)(void*);

	static void retire(
		oa::Engine& inEngine,
		void* inPayload,
		CompleteFn inComplete,
		ReleaseFn inRelease)
	{
		oa::EngineAccess(inEngine).retireBorrowedService(
			inPayload, inComplete, inRelease);
	}
};

} // namespace oa
