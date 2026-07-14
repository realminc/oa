#pragma once

#include <Oa/Core/Types.h>

class OaComputeGraph;
class OaEngine;
class OaGpuTimer;
class OaMatrix;
class OaComputeEngine;
class OaVkDevice;
class OaGraphicsEngine;
class OaVkQueue;
struct OaTexture;
struct OaSwapchain;

namespace OaRuntimeGlobal {
void SetRuntime(OaComputeEngine* InRuntime);
}

enum class OaContextMatMulPrecision : OaU8 {
	Auto,
	Fp32,
	Bf16,
};

struct OaClearColor {
	OaF32 R = 0.0F;
	OaF32 G = 0.0F;
	OaF32 B = 0.0F;
	OaF32 A = 1.0F;
};

struct OaRect2D {
	OaI32 X = 0;
	OaI32 Y = 0;
	OaI32 W = 0;
	OaI32 H = 0;

	[[nodiscard]] bool IsEmpty() const noexcept { return W <= 0 or H <= 0; }
};

struct OaBlitDesc {
	const OaTexture* Src     = nullptr;
	const OaTexture* Dst     = nullptr;
	OaFilter         Filter  = OaFilter::Linear;
	OaRect2D         SrcRect = {};
	OaRect2D         DstRect = {};
};
