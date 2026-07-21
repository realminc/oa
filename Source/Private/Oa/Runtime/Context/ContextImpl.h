#pragma once

#include "../ExecutionSession.h"

#include <Oa/Runtime/Sync.h>

class OaEngine;

// Private mutable state for the OaContext compatibility facade. Keeping this
// definition behind the public Context.h contract prevents execution and cache changes
// from recompiling every public Context.h consumer.
class OaContextImpl {
public:
	explicit OaContextImpl(OaEngine* InEngine)
		: Engine_(InEngine)
		, Execution_(InEngine) {}

	OaEngine* Engine_ = nullptr;
	OaExecutionSession Execution_;

};
