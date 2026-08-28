// TrainingProgram — fixed-shape training-step capture and replay.
//
// The program owns training-specific capture state and delegates the immutable
// compiled graph to the shared runtime ExecutionPlan. capture consumes the
// dispatches currently recorded by an engine only after compilation succeeds;
// subsequent replay bypasses forward/autograd/optimizer graph reconstruction
// entirely. dynamic eager training remains the fallback.

#pragma once

#include <oa/core/status.h>
#include <oa/core/matrix.h>
#include <oa/runtime/executionStats.h>
#include <oa/runtime/semanticBinding.h>
#include <oa/runtime/semanticGraph.h>
#include <oa/runtime/event.h>

namespace oa {

class ExecutableGraph;
class Engine;
class ExecutionPlan;

enum class TrainingCompilationStage : oa::U8 {
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

enum class TrainingCompilationState : oa::U8 {
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
struct TrainingCompilationStageRecord {
	TrainingCompilationStage stage =
		TrainingCompilationStage::SemanticValidation;
	TrainingCompilationState state = TrainingCompilationState::NotRun;
	oa::U32 inputCount = 0;
	oa::U32 outputCount = 0;
};

class TrainingProgramOptions {
public:
	// Emit the final compute-to-host visibility edge. Keep enabled when loss or
	// metrics are read on the CPU after every replay; disable for GPU-only chunks.
	oa::Bool hostReadbackRequired = true;

	// fail capture when a known host-mutated or nondeterministically frozen op is
	// present. This prevents a graph from silently reusing one dropout mask or a
	// push-constant optimizer step forever.
	oa::Bool validateReplaySafety = true;

	// Bracket the complete captured GPU program with timestamp queries. training
	// waits after each step, satisfying the graph's single-flight timing contract.
	oa::Bool enableGpuTiming = false;

	// values consumed outside the captured GPU program, such as the scalar loss
	// read by training metrics. The span is borrowed only for capture().
	oa::Span<const oa::Matrix* const> observedOutputs;
};

class TrainingProgram {
public:
	TrainingProgram();
	~TrainingProgram();

	TrainingProgram(const TrainingProgram&) = delete;
	TrainingProgram& operator=(const TrainingProgram&) = delete;
	TrainingProgram(TrainingProgram&&) = delete;
	TrainingProgram& operator=(TrainingProgram&&) = delete;

	// compile the engine's currently recorded fixed-shape step. On success the
	// source graph is cleared without execution and this program becomes its
	// sole executable owner. On failure the source graph is left intact so the
	// caller may execute it eagerly or diagnose the rejected node. The engine
	// and its vulkan device must outlive this program.
	[[nodiscard]] oa::Status capture(
		oa::Engine& inEngine,
		const TrainingProgramOptions& inOptions = {});

	// Non-blocking submit. Same-queue replays are ordered by vulkan; call wait()
	// before mapped host reads or resource mutation from the CPU.
	[[nodiscard]] oa::Status replay();
	[[nodiscard]] oa::Result<oa::Event> replayAsync();
	[[nodiscard]] oa::Status wait();

	// wait for pending work, release the compiled plan and all retained buffers.
	[[nodiscard]] oa::Status reset();

	[[nodiscard]] oa::Bool isCaptured() const noexcept;
	[[nodiscard]] oa::U32 nodeCount() const noexcept;
	[[nodiscard]] oa::U32 semanticOpCount() const noexcept {
		return semanticGraph_.operationCount();
	}
	[[nodiscard]] const oa::SemanticGraph& semanticGraph() const noexcept {
		return semanticGraph_;
	}
	[[nodiscard]] const oa::SemanticLoweringAnalysis& loweringAnalysis() const noexcept {
		return loweringAnalysis_;
	}
	[[nodiscard]] oa::Span<const oa::SemanticStorageBinding>
	semanticStorageBindings() const noexcept
	{
		return {semanticStorageBindings_.data(), semanticStorageBindings_.size()};
	}
	[[nodiscard]] oa::U32 capturedResourceCount() const noexcept {
		return static_cast<oa::U32>(capturedResourceOwners_.size());
	}
	[[nodiscard]] oa::Span<const oa::CapturedResourceDesc>
	capturedResources() const noexcept
	{
		return {capturedResources_.data(), capturedResources_.size()};
	}
	[[nodiscard]] oa::U32 aliasCandidateCount() const noexcept {
		return aliasCandidateCount_;
	}
	[[nodiscard]] oa::U32 plannedAliasGroupCount() const noexcept {
		return plannedAliasGroupCount_;
	}
	[[nodiscard]] oa::U64 potentialAliasSavings() const noexcept {
		return potentialAliasSavings_;
	}
	[[nodiscard]] oa::U64 materializedAliasSavings() const noexcept {
		return materializedAliasSavings_;
	}
	[[nodiscard]] oa::GraphStats stats() const;
	[[nodiscard]] oa::F64 lastGpuMs() const noexcept;
	[[nodiscard]] oa::String debugReportJson(
		oa::StringView inName = "TrainingStep") const;
	[[nodiscard]] oa::String semanticDebugReportJson(
		oa::StringView inName = "TrainingStep") const
	{
		return semanticGraph_.debugReportJson(inName);
	}
	[[nodiscard]] oa::Span<const TrainingCompilationStageRecord>
	compilationStages() const noexcept
	{
		return {compilationStages_.data(), compilationStages_.size()};
	}
	[[nodiscard]] oa::String compilationDebugReportJson(
		oa::StringView inName = "TrainingStep") const;

private:
	[[nodiscard]] static oa::Status validate_(const oa::ExecutableGraph& inGraph);
	[[nodiscard]] oa::Status prepareReplayRng_(oa::Engine& inRuntime);
	void recordCompilationStage_(
		TrainingCompilationStage inStage,
		TrainingCompilationState inState,
		oa::U32 inInputCount,
		oa::U32 inOutputCount);

	oa::UniquePtr<oa::ExecutionPlan> plan_;
	oa::SemanticGraph semanticGraph_;
	oa::SemanticLoweringAnalysis loweringAnalysis_;
	oa::Vector<oa::SemanticStorageBinding> semanticStorageBindings_;
	oa::Vector<oa::CapturedResourceDesc> capturedResources_;
	oa::Vector<oa::SharedPtr<oavk::Buffer>> capturedResourceOwners_;
	oa::U32 aliasCandidateCount_ = 0;
	oa::U32 aliasMaterializationEligibleCount_ = 0;
	oa::U32 plannedAliasGroupCount_ = 0;
	oa::U64 potentialAliasSavings_ = 0;
	oa::U64 materializedAliasSavings_ = 0;
	oa::String aliasMaterializationFallbackReason_;
	oa::Vector<TrainingCompilationStageRecord> compilationStages_;
	oa::Vector<oa::Matrix> rngStates_;
};

} // namespace oa
