// OaTrainingProgram — fixed-shape training-step capture and replay.
//
// The program owns training-specific capture state and delegates the immutable
// compiled graph to an internal OaExecutionPlan. Capture consumes the dispatches
// currently recorded in an OaContext only after compilation succeeds;
// subsequent replay bypasses forward/autograd/optimizer graph reconstruction
// entirely. Dynamic eager training remains the fallback.

#pragma once

#include <Oa/Core/Status.h>
#include <Oa/Core/Matrix.h>
#include <Oa/Runtime/ComputeGraph.h>
#include <Oa/Runtime/SemanticBinding.h>
#include <Oa/Runtime/SemanticGraph.h>

class OaContext;
class OaEngine;
class OaExecutionPlan;

enum class OaTrainingCompilationStage : OaU8 {
	SemanticValidation,
	ReplaySafety,
	Decomposition,
	Fusion,
	Placement,
	Precision,
	KernelSelection,
	LoweringValidation,
	MemoryPlanning,
	SynchronizationPlanning,
	CommandRecording,
};

enum class OaTrainingCompilationState : OaU8 {
	NotRun,
	Inherited,
	Analyzed,
	Applied,
	Failed,
};

// Stable, deterministic evidence for the current training-plan capture seam.
// `Inherited` means the decision still happened in the compatibility authoring
// or lowering path rather than in a unified graph compiler. `NotRun` is not a
// success: it makes an intentionally missing optimization visible.
struct OaTrainingCompilationStageRecord {
	OaTrainingCompilationStage Stage =
		OaTrainingCompilationStage::SemanticValidation;
	OaTrainingCompilationState State = OaTrainingCompilationState::NotRun;
	OaU32 InputCount = 0;
	OaU32 OutputCount = 0;
};

class OaTrainingProgramOptions {
public:
	// Emit the final compute-to-host visibility edge. Keep enabled when loss or
	// metrics are read on the CPU after every replay; disable for GPU-only chunks.
	OaBool HostReadbackRequired = true;

	// Fail capture when a known host-mutated or nondeterministically frozen op is
	// present. This prevents a graph from silently reusing one dropout mask or a
	// push-constant optimizer step forever.
	OaBool ValidateReplaySafety = true;

	// Bracket the complete captured GPU program with timestamp queries. Training
	// waits after each step, satisfying the graph's single-flight timing contract.
	OaBool EnableGpuTiming = false;

	// Values consumed outside the captured GPU program, such as the scalar loss
	// read by training metrics. The span is borrowed only for Capture().
	OaSpan<const OaMatrix* const> ObservedOutputs;
};

class OaTrainingProgram {
public:
	OaTrainingProgram();
	~OaTrainingProgram();

	OaTrainingProgram(const OaTrainingProgram&) = delete;
	OaTrainingProgram& operator=(const OaTrainingProgram&) = delete;
	OaTrainingProgram(OaTrainingProgram&&) = delete;
	OaTrainingProgram& operator=(OaTrainingProgram&&) = delete;

	// Compile the context's currently recorded fixed-shape step. On success the
	// source graph is cleared without execution and this program becomes its
	// sole executable owner. On failure the source graph is left intact so the
	// caller may execute it eagerly or diagnose the rejected node. The context's
	// compute runtime and Vulkan device must outlive this program.
	[[nodiscard]] OaStatus Capture(
		OaContext& InContext,
		const OaTrainingProgramOptions& InOptions = {});

	// Non-blocking submit. Same-queue replays are ordered by Vulkan; call Wait()
	// before mapped host reads or resource mutation from the CPU.
	[[nodiscard]] OaStatus Replay();
	[[nodiscard]] OaResult<OaCompletionToken> ReplayAsync();
	[[nodiscard]] OaStatus Wait();

	// Wait for pending work, release the compiled plan and all retained buffers.
	[[nodiscard]] OaStatus Reset();

	[[nodiscard]] OaBool IsCaptured() const noexcept;
	[[nodiscard]] OaU32 NodeCount() const noexcept;
	[[nodiscard]] OaU32 SemanticOperationCount() const noexcept {
		return SemanticGraph_.OperationCount();
	}
	[[nodiscard]] const OaSemanticGraph& SemanticGraph() const noexcept {
		return SemanticGraph_;
	}
	[[nodiscard]] const OaSemanticLoweringAnalysis& LoweringAnalysis() const noexcept {
		return LoweringAnalysis_;
	}
	[[nodiscard]] OaSpan<const OaSemanticStorageBinding>
	SemanticStorageBindings() const noexcept
	{
		return {SemanticStorageBindings_.Data(), SemanticStorageBindings_.Size()};
	}
	[[nodiscard]] OaU32 CapturedResourceCount() const noexcept {
		return static_cast<OaU32>(CapturedResourceOwners_.Size());
	}
	[[nodiscard]] OaSpan<const OaCapturedResourceDesc>
	CapturedResources() const noexcept
	{
		return {CapturedResources_.Data(), CapturedResources_.Size()};
	}
	[[nodiscard]] OaU32 AliasCandidateCount() const noexcept {
		return AliasCandidateCount_;
	}
	[[nodiscard]] OaU32 PlannedAliasGroupCount() const noexcept {
		return PlannedAliasGroupCount_;
	}
	[[nodiscard]] OaU64 PotentialAliasSavings() const noexcept {
		return PotentialAliasSavings_;
	}
	[[nodiscard]] OaU64 MaterializedAliasSavings() const noexcept {
		return MaterializedAliasSavings_;
	}
	[[nodiscard]] OaGraphStats Stats() const;
	[[nodiscard]] OaF64 LastGpuMs() const noexcept;
	[[nodiscard]] OaString DebugReportJson(
		OaStringView InName = "TrainingStep") const;
	[[nodiscard]] OaString SemanticDebugReportJson(
		OaStringView InName = "TrainingStep") const
	{
		return SemanticGraph_.DebugReportJson(InName);
	}
	[[nodiscard]] OaSpan<const OaTrainingCompilationStageRecord>
	CompilationStages() const noexcept
	{
		return {CompilationStages_.Data(), CompilationStages_.Size()};
	}
	[[nodiscard]] OaString CompilationDebugReportJson(
		OaStringView InName = "TrainingStep") const;

private:
	[[nodiscard]] static OaStatus Validate_(const OaComputeGraph& InGraph);
	[[nodiscard]] OaStatus PrepareReplayRng_(OaEngine& InRuntime);
	void RecordCompilationStage_(
		OaTrainingCompilationStage InStage,
		OaTrainingCompilationState InState,
		OaU32 InInputCount,
		OaU32 InOutputCount);

	OaUniquePtr<OaExecutionPlan> Plan_;
	OaSemanticGraph SemanticGraph_;
	OaSemanticLoweringAnalysis LoweringAnalysis_;
	OaVec<OaSemanticStorageBinding> SemanticStorageBindings_;
	OaVec<OaCapturedResourceDesc> CapturedResources_;
	OaVec<OaSharedPtr<OaVkBuffer>> CapturedResourceOwners_;
	OaU32 AliasCandidateCount_ = 0;
	OaU32 AliasMaterializationEligibleCount_ = 0;
	OaU32 PlannedAliasGroupCount_ = 0;
	OaU64 PotentialAliasSavings_ = 0;
	OaU64 MaterializedAliasSavings_ = 0;
	OaString AliasMaterializationFallbackReason_;
	OaVec<OaTrainingCompilationStageRecord> CompilationStages_;
	OaVec<OaMatrix> RngStates_;
};
