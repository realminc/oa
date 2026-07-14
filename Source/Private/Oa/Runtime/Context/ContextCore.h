#pragma once

#include "ContextTypes.h"
#include "FnContext.h"

#include <Oa/Core/Status.h>
#include <Oa/Core/BufferAccess.h>
#include <Oa/Runtime/Allocator.h>
#include <Oa/Runtime/Pool.h>
#include <Oa/Runtime/Sync.h>

#include <initializer_list>

class OaVideoEncoder;
class OaVideoDecoder;
struct OaVideoConversionOptions;
struct OaEncodedFrame;
struct OaVideoFrame;

class OaContext {
public:
	static OaContext* Create(OaEngine* InEngine);
	~OaContext();

	OaContext(const OaContext&) = delete;
	OaContext& operator=(const OaContext&) = delete;
	OaContext(OaContext&&) noexcept = delete;
	OaContext& operator=(OaContext&&) noexcept = delete;

	static void SetDefault(OaContext* InContext);
	[[nodiscard]] static OaContext& GetDefault();

	void Add(
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
		OaVkBuffer InA,
		OaVkBuffer InB,
		OaVkBuffer OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK
	);

	void AddMatMul(
		OaVkBuffer InA,
		OaVkBuffer InB,
		OaVkBuffer OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK,
		OaContextMatMulPrecision InPrecision
	);

	void AddMatMul(
		const OaMatrix& InA,
		const OaMatrix& InB,
		OaMatrix& OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK
	);

	void AddMatMul(
		const OaMatrix& InA,
		const OaMatrix& InB,
		OaMatrix& OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK,
		OaContextMatMulPrecision InPrecision
	);

	void AddLinear(
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
		const OaMatrix& InX,
		const OaMatrix& InWeight,
		const OaMatrix* InBias,
		OaMatrix& OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK
	);

	void AddLinearRelu(
		const OaMatrix& InX,
		const OaMatrix& InWeight,
		const OaMatrix& InBias,
		OaMatrix& OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK
	);

	void AddLinearGelu(
		const OaMatrix& InX,
		const OaMatrix& InWeight,
		const OaMatrix& InBias,
		OaMatrix& OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK
	);

	void AddLinearBwdWeightBias(
		const OaMatrix& InInput,
		const OaMatrix& InGradOutput,
		OaMatrix& OutGradWeight,
		OaMatrix& OutGradBias,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK
	);

	[[nodiscard]] OaTexture RecordAcquire(OaSwapchain& InSwap);
	void RecordPresent(OaSwapchain& InSwap, const OaTexture& InTarget);
	void RecordPresent(
		OaSwapchain& InSwap,
		const OaTexture& InTarget,
		const OaVkTimelineSemaphore& InWaitSemaphore,
		OaU64 InWaitValue);
	void SubmitPresent();
	void RecordBlit(const OaBlitDesc& InDesc);
	void RecordClear(const OaTexture& InTarget, OaClearColor InColor);
	void RecordImGuiFrame(const OaTexture& InTarget);
	[[nodiscard]] OaStatus RecordEncode(
		class OaVideoEncoder& InEncoder,
		const OaTexture& InRgba,
		OaU64 InPts,
		struct OaEncodedFrame& OutFrame);
	[[nodiscard]] OaStatus RecordDecode(
		class OaVideoDecoder& InDecoder,
		const OaSpan<const OaU8>& InAccessUnit,
		const struct OaVideoConversionOptions& InOptions,
		OaU64 InPts,
		struct OaVideoFrame& OutFrame);

	[[nodiscard]] OaStatus Execute();
	[[nodiscard]] OaStatus ExecuteAsync(OaGpuTimer* InTimer = nullptr);
	[[nodiscard]] OaResult<OaCompletionToken> ExecuteAsyncToken(OaGpuTimer* InTimer = nullptr);
	[[nodiscard]] OaStatus BeginAsyncBatch();
	[[nodiscard]] OaStatus ExecuteInAsyncBatch(OaGpuTimer* InTimer = nullptr);
	[[nodiscard]] OaStatus FlushAsyncBatch();
	[[nodiscard]] OaResult<OaCompletionToken> FlushAsyncBatchToken();
	[[nodiscard]] OaBool IsAsyncBatchActive() const noexcept;
	[[nodiscard]] OaStatus Sync();
	[[nodiscard]] OaU32 MaxAsyncSubmissions() const noexcept;
	void Clear();

	void SetTraining(bool InTraining) noexcept { Training_ = InTraining; }
	[[nodiscard]] bool IsTraining() const noexcept { return Training_; }

	class ScopedEval {
	public:
		explicit ScopedEval(OaContext& InCtx) : Ctx_(InCtx), Prev_(InCtx.IsTraining()) {
			InCtx.SetTraining(false);
		}
		~ScopedEval() { Ctx_.SetTraining(Prev_); }
		ScopedEval(const ScopedEval&) = delete;
		ScopedEval& operator=(const ScopedEval&) = delete;
		ScopedEval(ScopedEval&&) noexcept = delete;
		ScopedEval& operator=(ScopedEval&&) noexcept = delete;
	private:
		OaContext& Ctx_;
		bool Prev_;
	};

	class Scope {
	public:
		explicit Scope(OaContext& InCtx) : Ctx_(InCtx) {}
		~Scope() {
			(void)Ctx_.Execute();
			(void)Ctx_.Sync();
		}
		Scope(const Scope&) = delete;
		Scope& operator=(const Scope&) = delete;
		Scope(Scope&&) noexcept = delete;
		Scope& operator=(Scope&&) noexcept = delete;
	private:
		OaContext& Ctx_;
	};

	[[nodiscard]] OaEngine& Engine() const noexcept;
	[[nodiscard]] OaComputeEngine* VkCompute() const noexcept { return Runtime_; }
	[[nodiscard]] OaGraphicsEngine* VkGraphics() const noexcept;
	[[nodiscard]] bool HasCompute() const noexcept;
	[[nodiscard]] bool HasGraphics() const noexcept;
	[[nodiscard]] bool HasPresent() const noexcept;
	[[nodiscard]] bool HasMeshShader() const noexcept;
	[[nodiscard]] bool HasRayTrace() const noexcept;
	[[nodiscard]] bool IsRemote() const noexcept;
	[[nodiscard]] OaComputeEngine* GetRuntime() const noexcept { return Runtime_; }
	[[nodiscard]] OaComputeGraph* Graph() const noexcept { return Graph_; }
	[[nodiscard]] OaU32 NodeCount() const noexcept;

private:
	friend void OaRuntimeGlobal::SetRuntime(OaComputeEngine* InRuntime);
	friend void OaFnContext::Add(
		OaContext& InCtx,
		OaStringView InKernelName,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY,
		OaU32 InGroupsZ);
	friend void OaFnContext::Add(
		OaContext& InCtx,
		OaStringView InKernelName,
		std::initializer_list<const OaMatrix*> InMatrices,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY,
		OaU32 InGroupsZ);
	friend void OaFnContext::AddMatMul(
		OaContext& InCtx,
		OaVkBuffer InA,
		OaVkBuffer InB,
		OaVkBuffer OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);
	friend void OaFnContext::AddMatMul(
		OaContext& InCtx,
		OaVkBuffer InA,
		OaVkBuffer InB,
		OaVkBuffer OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK,
		OaContextMatMulPrecision InPrecision);
	friend void OaFnContext::AddMatMul(
		OaContext& InCtx,
		const OaMatrix& InA,
		const OaMatrix& InB,
		OaMatrix& OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);
	friend void OaFnContext::AddMatMul(
		OaContext& InCtx,
		const OaMatrix& InA,
		const OaMatrix& InB,
		OaMatrix& OutC,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK,
		OaContextMatMulPrecision InPrecision);
	friend void OaFnContext::AddLinear(
		OaContext& InCtx,
		OaVkBuffer InX,
		OaVkBuffer InWeight,
		OaVkBuffer InBias,
		OaVkBuffer OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK,
		OaBool InHasBias);
	friend void OaFnContext::AddLinear(
		OaContext& InCtx,
		const OaMatrix& InX,
		const OaMatrix& InWeight,
		const OaMatrix* InBias,
		OaMatrix& OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);
	friend void OaFnContext::AddLinearRelu(
		OaContext& InCtx,
		const OaMatrix& InX,
		const OaMatrix& InWeight,
		const OaMatrix& InBias,
		OaMatrix& OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);
	friend void OaFnContext::AddLinearGelu(
		OaContext& InCtx,
		const OaMatrix& InX,
		const OaMatrix& InWeight,
		const OaMatrix& InBias,
		OaMatrix& OutY,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);
	friend void OaFnContext::AddLinearBwdWeightBias(
		OaContext& InCtx,
		const OaMatrix& InInput,
		const OaMatrix& InGradOutput,
		OaMatrix& OutGradWeight,
		OaMatrix& OutGradBias,
		OaU32 InM,
		OaU32 InN,
		OaU32 InK);
	friend void OaFnContext::AddLinearActivation(
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
		OaU32 InK);
	friend void OaFnContext::AddOwned(
		OaContext& InCtx,
		OaStringView InKernelName,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaSharedPtr<OaVkBuffer>> InBufferOwners,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY,
		OaU32 InGroupsZ);

	explicit OaContext(OaComputeEngine* InRuntime);

	OaComputeEngine* Runtime_;
	OaComputeGraph* Graph_;
	OaBool Executed_ = false;
	bool Training_ = true;
	OaVec<OaComputeGraph*> DeferredGraphs_;
	OaVkQueue* ComputeQueue_ = nullptr;
	OaVkQueue* GraphicsQueue_ = nullptr;
	OaVkQueue* VideoQueue_ = nullptr;

	struct PendingPresent {
		OaSwapchain* Swap          = nullptr;
		void*        ImageHandle   = nullptr;
		OaU32        ImageIndex    = 0;
		OaU32        FrameSlot     = 0;
		bool         PresentQueued = false;
		bool         HasClear      = false;
		OaClearColor ClearColor;
		bool         HasBlit       = false;
		void*        BlitSrcBuffer = nullptr;
		void*        BlitSrcImage  = nullptr;
		OaI32        BlitSrcLayout = 0;
		OaU32        BlitSrcWidth  = 0;
		OaU32        BlitSrcHeight = 0;
		bool         HasImGuiDraw  = false;
		OaFilter     Filter        = OaFilter::Linear;
		void*        WaitTimelineSemaphore = nullptr;
		OaU64        WaitTimelineValue = 0;
	};
	PendingPresent PendingPresent_;

	void FlushPendingPresent();
};
