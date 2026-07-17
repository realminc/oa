#pragma once

#include <Oa/Core/Types.h>
#include <Oa/Runtime/Allocator.h>

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

enum class OaContextMatMulPrecision : OaU8 {
	Auto,
	Fp32,
	Bf16,
};

class OaContextExecutionStats {
public:
	OaU32 NodeCount = 0;
	OaU32 GraphCount = 0;
	OaU32 CompileCacheHits = 0;
	OaU32 BoundaryBarrierCount = 0;
	OaU32 HostBarrierCount = 0;
	OaF64 CompileMs = 0.0;
	OaF64 RecordMs = 0.0;
	OaF64 SubmitMs = 0.0;
	OaF64 WaitMs = 0.0;

	[[nodiscard]] OaF64 CpuMs() const noexcept {
		return CompileMs + RecordMs + SubmitMs + WaitMs;
	}
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

// Pending access state carried across secondary-command-buffer boundaries in a
// context-owned batch. The primary command buffer emits a barrier only for a
// buffer with a real cross-graph hazard; unrelated buffers remain pending until
// a later graph actually consumes or overwrites them.
class OaContextBatchBufferState {
public:
	OaVkBuffer Buffer;
	OaBool Read = false;
	OaBool Write = false;
	OaBool IndirectRead = false;
};
