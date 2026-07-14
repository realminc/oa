#pragma once

#include "ContextTypes.h"

#include <Oa/Core/Status.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Sync.h>

#include <initializer_list>

class OaContext;
class OaMatrix;

// Per-domain default-context accessors. These are the internal hooks used by
// OaFnMatrix, OaFnLoss, etc. to reach the thread-local default context.
namespace OaFnMatrix { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnLoss   { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnAudio  { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnUi     { [[nodiscard]] OaContext& GetContext(); }
namespace OaFnCrypto { [[nodiscard]] OaContext& GetContext(); }

namespace OaFnContext {

void Add(
	OaContext& InCtx,
	OaStringView InKernelName,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY = 1,
	OaU32 InGroupsZ = 1
);

void Add(
	OaContext& InCtx,
	OaStringView InKernelName,
	std::initializer_list<const OaMatrix*> InMatrices,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY = 1,
	OaU32 InGroupsZ = 1
);

void AddMatMul(
	OaContext& InCtx,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

void AddMatMul(
	OaContext& InCtx,
	OaVkBuffer InA,
	OaVkBuffer InB,
	OaVkBuffer OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaContextMatMulPrecision InPrecision
);

void AddMatMul(
	OaContext& InCtx,
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

void AddMatMul(
	OaContext& InCtx,
	const OaMatrix& InA,
	const OaMatrix& InB,
	OaMatrix& OutC,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaContextMatMulPrecision InPrecision
);

void AddLinear(
	OaContext& InCtx,
	OaVkBuffer InX,
	OaVkBuffer InWeight,
	OaVkBuffer InBias,
	OaVkBuffer OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK,
	OaBool InHasBias
);

void AddLinear(
	OaContext& InCtx,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix* InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

void AddLinearRelu(
	OaContext& InCtx,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

void AddLinearGelu(
	OaContext& InCtx,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

void AddLinearBwdWeightBias(
	OaContext& InCtx,
	const OaMatrix& InInput,
	const OaMatrix& InGradOutput,
	OaMatrix& OutGradWeight,
	OaMatrix& OutGradBias,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

// Internal helper used by Add / AddMatMul / AddLinear* to record a kernel with
// explicit buffer ownership for mixed-precision mirrors and matrix lifetimes.
void AddOwned(
	OaContext& InCtx,
	OaStringView InKernelName,
	OaSpan<OaVkBuffer> InBuffers,
	OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
	OaSpan<OaBufferAccess> InAccess,
	const void* InPush,
	OaU32 InPushSize,
	OaU32 InGroupsX,
	OaU32 InGroupsY = 1,
	OaU32 InGroupsZ = 1
);

// Internal helper used by AddLinearRelu / AddLinearGelu.
void AddLinearActivation(
	OaContext& InCtx,
	const char* InBf16SgName,
	const char* InBf16WgName,
	const char* InTiledName,
	const OaMatrix& InX,
	const OaMatrix& InWeight,
	const OaMatrix& InBias,
	OaMatrix& OutY,
	OaU32 InM,
	OaU32 InN,
	OaU32 InK
);

} // namespace OaFnContext
