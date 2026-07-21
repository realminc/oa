#pragma once

#include <Oa/Runtime/Engine.h>

class OaBorrowedServiceRetirement {
public:
	using CompleteFn = OaStatus (*)(void*);
	using ReleaseFn = void (*)(void*);

	static void Retire(
		OaEngine& InEngine,
		void* InPayload,
		CompleteFn InComplete,
		ReleaseFn InRelease)
	{
		InEngine.RetireBorrowedService_(
			InPayload, InComplete, InRelease);
	}
};
