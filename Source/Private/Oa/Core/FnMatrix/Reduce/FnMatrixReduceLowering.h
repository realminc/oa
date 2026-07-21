#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>

class OaContext;
class OaMatrix;
class OaOperationContract;

namespace OaFnMatrixPrivate {

// Lower a full arithmetic mean into Sum + Scale without creating a second
// semantic operation. The caller owns output allocation and semantic recording;
// both executable nodes retain that caller's operation identity and provenance.
[[nodiscard]] OaStatus LowerFullMean(
	OaContext& InContext,
	const OaMatrix& InInput,
	OaMatrix& OutMean,
	const OaOperationContract& InContract,
	OaU32 InSemanticOperation
);

} // namespace OaFnMatrixPrivate
