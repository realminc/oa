#pragma once

#include <Oa/Core/Types.h>

using OaSemanticValueId = OaU32;
using OaSemanticOperationId = OaU32;

constexpr OaSemanticValueId OaInvalidSemanticValueId = UINT32_MAX;
constexpr OaSemanticOperationId OaInvalidSemanticOperationId = UINT32_MAX;

class OaSemanticGraph;
