#pragma once

#include <oa/core/status.h>
#include <oa/runtime/semanticGraphFwd.h>

namespace oa {

class ExecutionSession;
class Matrix;

struct LinearWeightBiasBwdGraphDesc {
	const Matrix* input = nullptr;
	const Matrix* gradOutput = nullptr;
	Matrix* gradWeight = nullptr;
	Matrix* gradBias = nullptr;
	StringView operation = {};
	U64 opContractHash = 0;
	U32 semanticOp = invalidSemanticOpId;
};

// Private DNN semantic-to-executable lowering. Public ML operations describe
// values and provenance; this owner validates exact shapes and chooses a
// concrete engine without adding neural-network methods to oa::ExecutionSession.
class DnnGraphLowering {
public:
	[[nodiscard]] static Status recordLinearWeightBiasBackward(
		ExecutionSession& inContext,
		const LinearWeightBiasBwdGraphDesc& inDesc
	);
};

} // namespace oa
