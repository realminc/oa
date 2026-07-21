#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Runtime/SemanticGraphFwd.h>

class OaContext;
class OaMatrix;

struct OaLinearWeightBiasBwdGraphDesc {
	const OaMatrix* Input = nullptr;
	const OaMatrix* GradOutput = nullptr;
	OaMatrix* GradWeight = nullptr;
	OaMatrix* GradBias = nullptr;
	OaStringView Operation = {};
	OaU64 OperationContractHash = 0;
	OaSemanticOperationId SemanticOperation = OaInvalidSemanticOperationId;
};

// Private DNN semantic-to-executable lowering. Public ML operations describe
// values and provenance; this owner validates exact shapes and chooses a
// concrete engine without adding neural-network methods to OaContext.
class OaDnnGraphLowering {
public:
	[[nodiscard]] static OaStatus RecordLinearWeightBiasBackward(
		OaContext& InContext,
		const OaLinearWeightBiasBwdGraphDesc& InDesc);
};
