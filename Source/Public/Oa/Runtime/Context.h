#pragma once

#include <Oa/Core/BufferAccess.h>
#include <Oa/Core/Status.h>
#include <Oa/Core/Types.h>
#include <Oa/Runtime/SemanticGraphFwd.h>
#include <Oa/Runtime/Sync.h>

#include <initializer_list>

class OaComputeDispatchDesc;
class OaComputeGraph;
class OaContextImpl;
class OaExecutionMemory;
class OaEngine;
class OaGpuTimer;
class OaMatrix;
class OaMatrixDispatchDesc;
class OaOperationAttribute;
class OaOperationContract;
class OaSemanticGraph;
class OaVkBuffer;

// Stable execution telemetry exposed after an explicit context boundary.
// Mutable graph/session state remains private behind OaContextImpl.
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

// Compatibility facade for semantic recording and explicit execution. The
// declaration is public and self-contained; graph/session/lowering state lives
// exclusively in Source/Private behind OaContextImpl.
class OaContext {
public:
	static OaContext* Create(OaEngine* InEngine);
	~OaContext();

	OaContext(const OaContext&) = delete;
	OaContext& operator=(const OaContext&) = delete;
	OaContext(OaContext&&) noexcept = delete;
	OaContext& operator=(OaContext&&) noexcept = delete;

	static void SetDefault(OaContext* InContext);
	[[nodiscard]] static OaContext* GetDefaultPtr() noexcept;
	[[nodiscard]] static OaContext& GetDefault();

	// Canonical semantic-to-runtime boundary. Descriptors are validated and
	// copied into the active graph before these calls return.
	[[nodiscard]] OaStatus Record(const OaComputeDispatchDesc& InDesc);
	[[nodiscard]] OaStatus Record(const OaMatrixDispatchDesc& InDesc);
	[[nodiscard]] OaResult<OaSemanticOperationId> RecordOperation(
		const OaOperationContract& InContract,
		std::initializer_list<const OaMatrix*> InInputs,
		std::initializer_list<const OaMatrix*> InOutputs,
		std::initializer_list<OaOperationAttribute> InAttributes = {});

	void Add(
		OaStringView InKernelName,
		OaSpan<OaVkBuffer> InBuffers,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1,
		OaStringView InOperation = {},
		OaU64 InImplementationId = 0,
		OaU64 InOperationContractHash = 0,
		OaU64 InKernelContentHash = 0,
		OaU64 InProblemContractHash = 0,
		OaSemanticOperationId InSemanticOperation = OaInvalidSemanticOperationId);

	void Add(
		OaStringView InKernelName,
		std::initializer_list<const OaMatrix*> InMatrices,
		OaSpan<OaBufferAccess> InAccess,
		const void* InPush,
		OaU32 InPushSize,
		OaU32 InGroupsX,
		OaU32 InGroupsY = 1,
		OaU32 InGroupsZ = 1,
		OaStringView InOperation = {},
		OaU64 InImplementationId = 0,
		OaU64 InOperationContractHash = 0,
		OaU64 InKernelContentHash = 0,
		OaU64 InProblemContractHash = 0,
		OaSemanticOperationId InSemanticOperation = OaInvalidSemanticOperationId);

	[[nodiscard]] OaStatus Execute();
	// Submit never waits; Wait is the corresponding host boundary and reclaims
	// completed graph state.
	[[nodiscard]] OaResult<OaEvent> Submit(OaGpuTimer* InTimer = nullptr);
	[[nodiscard]] OaStatus Wait(const OaEvent& InEvent);
	[[nodiscard]] OaStatus BeginAsyncBatch();
	[[nodiscard]] OaStatus ExecuteInAsyncBatch(OaGpuTimer* InTimer = nullptr);
	// Submit the batch owned by this context and return its exact completion.
	// One batch may be in flight per context until Wait() consumes the event.
	[[nodiscard]] OaResult<OaEvent> SubmitBatch();
	[[nodiscard]] OaBool IsAsyncBatchActive() const noexcept;
	// True only after queue submission has accepted a batch and before its
	// exact completion has been consumed or retired. This lets composed
	// sessions distinguish a pre-submit rollback from accepted-but-unobservable
	// work when an internal completion-handle invariant fails.
	[[nodiscard]] OaBool HasPendingSubmission() const noexcept;
	[[nodiscard]] OaStatus Sync();
	[[nodiscard]] OaU32 MaxAsyncSubmissions() const noexcept;
	void Clear();

	// Selects the active recorder without submitting or waiting in its
	// destructor. Submission and host observation are always explicit.
	class RecordingScope {
	public:
		explicit RecordingScope(OaContext& InContext)
			: Previous_(OaContext::GetDefaultPtr()) {
			OaContext::SetDefault(&InContext);
		}
		~RecordingScope() { OaContext::SetDefault(Previous_); }

		RecordingScope(const RecordingScope&) = delete;
		RecordingScope& operator=(const RecordingScope&) = delete;
		RecordingScope(RecordingScope&&) noexcept = delete;
		RecordingScope& operator=(RecordingScope&&) noexcept = delete;

	private:
		OaContext* Previous_ = nullptr;
	};

	[[nodiscard]] OaEngine& Engine() const noexcept;
	[[nodiscard]] OaEngine* VkCompute() const noexcept;
	[[nodiscard]] bool HasCompute() const noexcept;
	[[nodiscard]] OaEngine* GetEngine() const noexcept;
	[[nodiscard]] OaComputeGraph* Graph() const noexcept;
	[[nodiscard]] OaSemanticGraph* SemanticGraph() const noexcept;
	[[nodiscard]] OaU32 NodeCount() const noexcept;
	[[nodiscard]] const OaContextExecutionStats& LastExecutionStats() const noexcept;

private:
	friend class OaExecutionMemory;
	explicit OaContext(OaEngine* InEngine);

	OaContextImpl* Impl_ = nullptr;
};
