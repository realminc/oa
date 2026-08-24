#pragma once

#include <oa/core/status.h>
#include <oa/core/types.h>

namespace oa { class ExecutionSession; }
namespace oa { class Matrix; }
namespace oa { class OpContract; }

namespace oa {

namespace FnMatrixPrivate {

// Lower a full arithmetic mean into Sum + scale without creating a second
// semantic operation. The caller owns output allocation and semantic recording;
// both executable nodes retain that caller's operation identity and provenance.
[[nodiscard]] oa::Status lowerFullMean(
	oa::ExecutionSession& inContext,
	const oa::Matrix& inInput,
	oa::Matrix& outMean,
	const oa::OpContract& inContract,
	oa::U32 inSemanticOp
);

} // namespace FnMatrixPrivate

} // namespace oa
